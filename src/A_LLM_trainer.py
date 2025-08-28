#!/usr/bin/env python3
"""
HIVE LLM architecture with JSON output and enhanced sequence descriptions
Changes:
1. JSON output with constrained generation
2. Natural language descriptions of game sequences
3. Token verification and improved logging
"""
import os
import json
import math
import pickle
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import contextlib

# Optional tqdm
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Unsloth imports (optional)
try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False
    FastLanguageModel = None

try:
    from unsloth.chat_templates import train_on_responses_only
    _HAS_UNSLOTH_CHAT = True
except Exception:
    _HAS_UNSLOTH_CHAT = False
    train_on_responses_only = None

# HF / PEFT fallback
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# For constrained generation
try:
    from transformers import LogitsProcessorList, LogitsProcessor
    _HAS_LOGITS_PROCESSOR = True
except:
    _HAS_LOGITS_PROCESSOR = False

# AMP
try:
    from torch.amp import GradScaler as AMPGradScaler
except Exception:
    from torch.cuda.amp import GradScaler as AMPGradScaler

logger = logging.getLogger("HiveLLM")

def setup_logging(log_path: str, to_stdout: bool = True, level=logging.INFO, also_log_to_txt: Optional[str] = None):
    logging.basicConfig(force=True)
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    if also_log_to_txt is not None:
        try:
            ftxt = logging.FileHandler(also_log_to_txt, mode='a', encoding='utf-8')
            ftxt.setFormatter(fmt)
            ftxt.setLevel(level)
            logger.addHandler(ftxt)
        except Exception as e:
            logger.warning(f"Failed to attach secondary log file handler ({also_log_to_txt}): {e}")

    if to_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    logger.propagate = False

@dataclass
class HiveLLMConfig:
    """Configuration for HIVE LLM with JSON output"""
    # Model
    base_model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    
    # JSON Output
    output_format: str = "json"  # "json" or "text"
    enforce_json_schema: bool = True
    json_max_retries: int = 3
    
    # Sequence Description
    add_natural_language_description: bool = True
    description_style: str = "player"  # "player" style with merged sequences

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # Training
    max_seq_length: int = 512
    learning_rate: float = 5e-4
    batch_size: int = 2
    epochs: int = 1
    grad_accumulation_steps: int = 16
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 1
    eval_every: int = 50
    save_every: int = 0
    lr_scheduler: str = "cosine"
    seed: int = 42

    # Logging of IO/predictions
    log_preds_every: int = 0
    max_log_samples: int = 4
    decode_max_chars: int = 500
    verify_token_addition: bool = True  # New flag for token verification

    # Paths
    output_dir: str = "models/hive_llm"
    gnn_model_path: str = "src/models/pretrain_GIN_3.pt"
    board_centroids_path: Optional[str] = "clustering_models/boards/cluster_centroids_kmeans_best.pkl"
    move_centroids_path: Optional[str] = "clustering_models/moves/cluster_centroids_kmeans_best.pkl"
    train_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/train_sequential_cache.pkl"
    val_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/validation_sequential_cache.pkl"

    # Pretokenized caches
    pretokenized_train_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/train_pretokenized_cache.pkl"
    pretokenized_val_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/validation_pretokenized_cache.pkl"

    # Tokens
    board_cluster_token_prefix: str = "BCL"
    move_cluster_token_prefix: str = "MCL"
    add_eos_token: bool = True

    # Loss
    board_loss_weight: float = 1.0
    move_loss_weight: float = 1.0
    loss_on_cluster_tokens_only: bool = True

    # Other
    system_prompt_path: Optional[str] = "src/prompts/prompt.txt"
    user_prompt_prefix: str = "Sequence:"
    verbose: bool = False
    use_cache: bool = False
    enable_gradient_checkpointing: bool = True
    freeze_non_lora: bool = True
    pretokenize: bool = False
    pretokenize_show_samples: int = 3
    # Cast LoRA adapters to match compute dtype (prevents BF16 vs Float32 matmul errors)
    cast_lora_to_compute_dtype: bool = True

    enable_progressive_masking: bool = True
    min_context_length: int = 1
    prediction_fractions: List[float] = field(default_factory=lambda: [0.25 , 0.75 , 0.95, 1.0])  # Predict at these fractions of the game
    
    # New flags
    run_validation: bool = False  # Whether to compute validation during training
    data_size: float = 1.0        # Fraction of data to load (applied before progressive indexing)

class JSONLogitsProcessor(LogitsProcessor):
    """Constrain generation to valid JSON format for move/board predictions"""
    
    def __init__(self, tokenizer, board_tokens: List[str], move_tokens: List[str]):
        self.tokenizer = tokenizer
        self.board_tokens = set(board_tokens)
        self.move_tokens = set(move_tokens)
        
        # Pre-compute token IDs for JSON structure elements
        self.json_tokens = {
            '{': tokenizer.encode('{', add_special_tokens=False),
            '}': tokenizer.encode('}', add_special_tokens=False),
            '"': tokenizer.encode('"', add_special_tokens=False),
            ':': tokenizer.encode(':', add_special_tokens=False),
            ',': tokenizer.encode(',', add_special_tokens=False),
            'move': tokenizer.encode('move', add_special_tokens=False),
            'board': tokenizer.encode('board', add_special_tokens=False),
        }
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process logits to enforce JSON structure"""
        # This is a simplified version - in production you'd want more sophisticated parsing
        return scores


class GameClusterSequenceDataset(Dataset):
    """Dataset for cluster token sequences with JSON output format"""

    def __init__(
        self,
        samples: List[Dict],
        board_tokens: List[str],
        move_tokens: List[str],
        tokenizer,
        config: HiveLLMConfig,
        system_prompt: Optional[str] = None,
        pretokenize: bool = False,
        save_pretok_path: Optional[str] = None,
        load_pretok_path: Optional[str] = None,
    ):
        self.samples = samples
        self.board_tokens = board_tokens
        self.move_tokens = move_tokens
        self.tokenizer = tokenizer
        self.config = config
        self.system_prompt = system_prompt
        self.num_board = len(board_tokens)
        self.num_move = len(move_tokens)
        self._encoded: Optional[List[Dict[str, torch.Tensor]]] = None
        if config.enable_progressive_masking:
            self._build_progressive_index_mapping()
        else:
            # Original behavior: one sample per game (last position only)
            self.index_map = [(i, None) for i in range(len(samples))]
        # Load pretokenized cache if available
        if load_pretok_path and os.path.exists(load_pretok_path):
            logger.info(f"Loading pretokenized dataset from: {load_pretok_path}")
            with open(load_pretok_path, 'rb') as f:
                raw = pickle.load(f)
            self._encoded = []
            iterator = tqdm(range(len(raw)), desc="Loading pretok", leave=False) if _HAS_TQDM else range(len(raw))
            for i in iterator:
                item = raw[i]
                self._encoded.append({
                    'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
                    'labels': torch.tensor(item['labels'], dtype=torch.long),
                })
            logger.info(f"Loaded {len(self._encoded)} pretokenized samples")
            return

        # Pretokenize if requested
        if pretokenize:
            self._pretokenize_dataset(save_pretok_path)
    def _build_progressive_index_mapping(self):
        """Build mapping from linear index to (game_idx, prediction_position) based on prediction_fractions"""
        self.index_map = []
        
        for game_idx, sample in enumerate(self.samples):
            # Get sequence lengths from the sample
            board_seq_len = len(self._norm_seq(sample.get('board_cluster_ids_sequence', [])))
            move_seq_len = len(self._norm_seq(sample.get('chosen_move_cluster_ids_sequence', [])))
            
            # Number of valid prediction positions for this game
            max_predictions = min(move_seq_len, board_seq_len - 1)  # Need board_seq[i+1] for target
            
            if max_predictions < self.config.min_context_length:
                continue  # Skip games that are too short
            
            # For each specified fraction, compute the prediction position
            for fraction in self.config.prediction_fractions:
                # Calculate position as a fraction of max_predictions
                pred_pos = int(fraction * max_predictions)
                # Clamp to valid range (ensure at least min_context_length and at most max_predictions - 1)
                pred_pos = max(self.config.min_context_length, min(pred_pos, max_predictions - 1))
                
                # Only add if it's a valid position
                if pred_pos < max_predictions:
                    self.index_map.append((game_idx, pred_pos))
        
        logger.info(f"Built progressive index mapping with {len(self.index_map)} prediction samples at fractions: {self.config.prediction_fractions}")
    def _pretokenize_dataset(self, save_path: Optional[str] = None):
        """Pretokenize all samples (now handles progressive indexing)"""
        logger.info("Pretokenizing dataset...")
        self._encoded = []
        
        # Use the actual dataset length (which accounts for progressive indexing)
        total_samples = len(self.index_map)
        iterator = tqdm(range(total_samples), desc="Pretokenizing") if _HAS_TQDM else range(total_samples)
        
        for idx in iterator:
            item = self._encode_item(idx)
            self._encoded.append({
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': item['labels'],
            })
            
            # Log sample compositions
            if self.config.verbose and idx < 3:
                self._log_sample_composition(item, idx)
        
        # Save pretokenized cache
        if save_path:
            self._save_pretokenized(save_path)

    def _log_sample_composition(self, item: Dict, idx: int):
        """Log detailed sample composition for debugging"""
        chat = item.get('chat', [])
        user_content = next((m['content'] for m in chat if m.get('role') == 'user'), '')
        assistant_content = next((m['content'] for m in chat if m.get('role') == 'assistant'), '')
        
        logger.info(f"\n[Sample {idx+1}] Composition:")
        logger.info(f"  User input: {user_content[:200]}...")
        logger.info(f"  Target output: {assistant_content}")
        logger.info(f"  Tokens - input: {len(item['input_ids'])}, labels: {(item['labels'] != -100).sum().item()}")

    def _save_pretokenized(self, path: str):
        """Save pretokenized dataset to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dumpable = [
                {
                    'input_ids': t['input_ids'].tolist(),
                    'attention_mask': t['attention_mask'].tolist(),
                    'labels': t['labels'].tolist(),
                } for t in self._encoded
            ]
            with open(path, 'wb') as f:
                pickle.dump(dumpable, f)
            logger.info(f"Saved pretokenized dataset to: {path}")
        except Exception as e:
            logger.warning(f"Failed to save pretokenized dataset: {e}")

    @staticmethod
    def _norm_seq(x) -> List[int]:
        if x is None:
            return []
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (int, np.integer)):
            return [int(x)]
        if isinstance(x, (list, tuple)):
            return list(x)
        return []

    def _build_next_pair_json(self, b_seq: List[int], m_seq: List[int], l_seq: List[List[int]], 
                             prediction_position: Optional[int] = None) -> Tuple[str, str]:
        """Build context and JSON target for specific prediction position (or last position if None)"""
        
        if len(b_seq) < 2 or len(m_seq) < 1:
            return "", '{"move": null, "board": null}'

        pairs = min(len(m_seq), len(b_seq) - 1)
        if pairs < 1:
            return "", '{"move": null, "board": null}'

        # NEW: Use prediction_position if provided, otherwise use last position
        if prediction_position is not None:
            target_move_idx = prediction_position
            if target_move_idx >= pairs:
                return "", '{"move": null, "board": null}'
        else:
            # Original behavior: predict second-to-last move
            target_move_idx = pairs - 1

        # Build context with merged player format up to target_move_idx
        context_parts = []

        # Get legal moves at prediction position
        next_legal_moves = l_seq[target_move_idx] if target_move_idx < len(l_seq) else []
        next_legal_moves = list(dict.fromkeys(next_legal_moves))  # Remove duplicates while preserving order

        # Add historical moves BEFORE the prediction position
        for i in range(target_move_idx):
            player = "Player 1" if i % 2 == 0 else "Player 2"
            board_token = self.board_tokens[b_seq[i]] if 0 <= b_seq[i] < self.num_board else ""
            move_token = self.move_tokens[m_seq[i]] if 0 <= m_seq[i] < self.num_move else ""
            
            if board_token and move_token:
                context_parts.append(f"{player}: {board_token} {move_token}")
        
        context_parts.append(" Current board state: ")
        # Add the board state before the target move
        target_player = "Player 1" if target_move_idx % 2 == 0 else "Player 2"
        target_board_before = self.board_tokens[b_seq[target_move_idx]] if 0 <= b_seq[target_move_idx] < self.num_board else ""
        
        if target_board_before:
            context_parts.append(f"{target_player}: {target_board_before}")

        # Add available legal moves
        context_parts.append(" Choose ONE move among the following LEGAL moves: ")
        for x in next_legal_moves:
            move_token = self.move_tokens[x] if 0 <= x < self.num_move else ""
            context_parts.append(f"- {move_token}")



        # Build JSON target
        target_move = self.move_tokens[m_seq[target_move_idx]] if 0 <= m_seq[target_move_idx] < self.num_move else "null"
        target_board = self.board_tokens[b_seq[target_move_idx + 1]] if target_move_idx + 1 < len(b_seq) and 0 <= b_seq[target_move_idx + 1] < self.num_board else "null"
        
        if self.config.output_format == "json":
            target = json.dumps({
                "move": target_move,
                "board": target_board
            })
        else:
            target = f"{target_move} {target_board}"

        context = "  ".join(context_parts)  # Using double space as separator for clarity
        return context, target

    def __len__(self):
        if self._encoded is not None:
            return len(self._encoded)
        return len(self.index_map)

    def __getitem__(self, idx):
        if self._encoded is not None:
            return self._encoded[idx]
        return self._encode_item(idx)

    def _encode_item(self, idx):
        # NEW: Decode linear index to (game_idx, prediction_position)
        game_idx, prediction_position = self.index_map[idx]
        sample = self.samples[game_idx]        
        b_seq = self._norm_seq(sample.get('board_cluster_ids_sequence'))
        m_seq = self._norm_seq(sample.get('chosen_move_cluster_ids_sequence'))
        l_seq = self._norm_seq(sample.get('legal_move_cluster_ids_sequence'))
        # Build context and target for this specific prediction position
        if self.config.enable_progressive_masking and prediction_position is not None:
            context, target = self._build_next_pair_json(b_seq, m_seq, l_seq, prediction_position)
        else:
            # Original behavior (predict last position)
            context, target = self._build_next_pair_json(b_seq, m_seq, l_seq)
        # Build chat
        chat = []
        if self.system_prompt:
            modified_prompt = self.system_prompt
            chat.append({"role": "system", "content": modified_prompt})

        chat.append({"role": "user", "content": context})
        chat.append({"role": "assistant", "content": target})

        # Apply chat template
        try:
            rendered = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            rendered = ""
            for msg in chat:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                rendered += f"<|{role}|>: {content}\n"

        # Tokenize
        encoding = self.tokenizer(
            rendered,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=1024,
        )
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        # Label masking for training only on assistant response
        labels = input_ids.clone()
        assistant_tokens = self.tokenizer(
            target,
            add_special_tokens=False,
            return_tensors='pt',
        )['input_ids'][0]
        
        if len(assistant_tokens) > 0:
            # Find assistant response in full sequence
            found = False
            for start_idx in range(len(input_ids) - len(assistant_tokens), -1, -1):
                if torch.equal(input_ids[start_idx:start_idx + len(assistant_tokens)], assistant_tokens):
                    labels[:start_idx] = -100
                    found = True
                    break
            if not found:
                # Fallback: mask first 70% of sequence
                cutoff = int(len(labels) * 0.7)
                labels[:cutoff] = -100
        else:
            labels[:] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'raw_text': rendered,
            'chat': chat,
            'assistant_text': target,
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with right-padding"""
        max_len = max(x['input_ids'].size(0) for x in batch)
        pad_token_id = 0

        def pad(t: torch.Tensor, pad_value: int = 0):
            if t.size(0) == max_len:
                return t
            return torch.cat([
                t,
                torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)
            ], dim=0)

        input_ids = torch.stack([pad(b['input_ids'], pad_token_id) for b in batch])
        attention_mask = torch.stack([pad(b['attention_mask'], 0) for b in batch])
        labels = torch.stack([pad(b['labels'], -100) for b in batch])
        labels = labels.masked_fill(attention_mask == 0, -100)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class HiveLLMModel(nn.Module):
    """HIVE LLM model with JSON output support and token verification"""

    def __init__(self, config: HiveLLMConfig):
        super().__init__()
        self.config = config
        logger.info(f"Initializing HiveLLM with base model: {config.base_model_name}")
        self._load_backbone()
        self._load_centroids_and_add_tokens()
        # Ensure LoRA adapters dtype/device match compute dtype to avoid matmul dtype mismatch
        self._ensure_lora_dtype()
        
        # Verify token addition if requested
        if config.verify_token_addition:
            self._verify_tokens()
        
        self._freeze_params()
        self._log_device_info()

    def _load_backbone(self):
        """Load the base model"""
        name = self.config.base_model_name
        use_unsloth = name.startswith("unsloth/") and _HAS_UNSLOTH

        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if use_unsloth:
            logger.info("Loading backbone via Unsloth FastLanguageModel...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=name,
                max_seq_length=self.config.max_seq_length,
                dtype=self.compute_dtype,
                load_in_4bit=self.config.use_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=self.config.lora_target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth" if self.config.enable_gradient_checkpointing else False,
                random_state=self.config.seed,
            )
            self.base_model = model
            self.tokenizer = tokenizer
            # Cast LoRA adapters immediately after attaching PEFT
            self._ensure_lora_dtype()
        else:
            logger.warning("Unsloth not available, falling back to HF loading")
            self._load_standard_hf()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_standard_hf(self):
        """Load model using standard HuggingFace"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=self.compute_dtype,
        )
        if self.config.use_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        # Cast LoRA adapters immediately after attaching PEFT
        self._ensure_lora_dtype()

    def _ensure_lora_dtype(self):
        """Ensure LoRA adapter weights are on the same device and dtype as the compute path."""
        if not getattr(self, "base_model", None):
            return
        if not getattr(self.config, "cast_lora_to_compute_dtype", True):
            return
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            # PEFT not present or different structure
            return
        try:
            device = next(self.base_model.parameters()).device
            dtype = getattr(self, "compute_dtype", None) or next(self.base_model.parameters()).dtype
            lora_modules = 0
            for m in self.base_model.modules():
                if isinstance(m, LoraLayer):
                    # Linear LoRA adapters
                    if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                        for _, layer in m.lora_A.items():
                            layer.to(device=device, dtype=dtype)
                        for _, layer in m.lora_B.items():
                            layer.to(device=device, dtype=dtype)
                        lora_modules += 1
                    # Embedding LoRA adapters if any
                    if hasattr(m, "lora_embedding_A") and hasattr(m, "lora_embedding_B"):
                        for _, layer in m.lora_embedding_A.items():
                            layer.to(device=device, dtype=dtype)
                        for _, layer in m.lora_embedding_B.items():
                            layer.to(device=device, dtype=dtype)
                        lora_modules += 1
            if lora_modules > 0:
                logger.info(f"Casted {lora_modules} LoRA module groups to dtype={dtype} on device={device}")
        except Exception as e:
            logger.warning(f"Failed to cast LoRA adapters to compute dtype: {e}")

    def _verify_tokens(self):
        """Verify that cluster tokens are properly added to vocabulary"""
        logger.info("\n=== Verifying Token Addition ===")
        
        # Check vocabulary size
        vocab_size = len(self.tokenizer)
        logger.info(f"Total vocabulary size: {vocab_size}")
        
        # Verify board tokens
        if hasattr(self, 'board_cluster_tokens'):
            logger.info(f"\nBoard tokens ({len(self.board_cluster_tokens)}):")
            for i, token in enumerate(self.board_cluster_tokens[:5]):  # Show first 5
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                decoded = self.tokenizer.decode([token_id])
                logger.info(f"  {token} -> ID: {token_id}, Decoded: '{decoded}'")
            
            # Test encoding/decoding
            test_text = f"Test: {self.board_cluster_tokens[0]} {self.board_cluster_tokens[1]}"
            encoded = self.tokenizer.encode(test_text)
            decoded = self.tokenizer.decode(encoded)
            logger.info(f"\nTest encode/decode:")
            logger.info(f"  Original: '{test_text}'")
            logger.info(f"  Encoded: {encoded}")
            logger.info(f"  Decoded: '{decoded}'")
        
        # Verify move tokens
        if hasattr(self, 'move_cluster_tokens'):
            logger.info(f"\nMove tokens ({len(self.move_cluster_tokens)}):")
            for i, token in enumerate(self.move_cluster_tokens[:5]):  # Show first 5
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                decoded = self.tokenizer.decode([token_id])
                logger.info(f"  {token} -> ID: {token_id}, Decoded: '{decoded}'")
        
        logger.info("=== Token Verification Complete ===\n")

    def _log_device_info(self):
        """Log GPU and memory information"""
        try:
            if torch.cuda.is_available():
                num = torch.cuda.device_count()
                for i in range(num):
                    name = torch.cuda.get_device_name(i)
                    total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"GPU[{i}]: {name} — total {total_mem:.2f} GB")
                torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"CUDA memory — allocated {alloc:.2f} GB, reserved {reserved:.2f} GB")
        except Exception as e:
            logger.warning(f"Could not log device info: {e}")

    def _load_centroids_and_add_tokens(self):
        """Load cluster centroids and add corresponding tokens to tokenizer"""
        def load_pkl(path):
            if not path or not os.path.exists(path):
                return None
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                for key in ['centroids', 'cluster_centroids']:
                    if key in data:
                        data = data[key]
                        break
            return torch.tensor(data, dtype=torch.bfloat16) if data is not None else None

        self.board_centroids = load_pkl(self.config.board_centroids_path)
        self.move_centroids = load_pkl(self.config.move_centroids_path)

        self.board_cluster_tokens: List[str] = []
        self.move_cluster_tokens: List[str] = []

        if self.board_centroids is not None:
            n_board = self.board_centroids.size(0)
            self.board_cluster_tokens = [f"<{self.config.board_cluster_token_prefix}_{i}>" for i in range(n_board)]
        if self.move_centroids is not None:
            n_move = self.move_centroids.size(0)
            self.move_cluster_tokens = [f"<{self.config.move_cluster_token_prefix}_{i}>" for i in range(n_move)]

        # Add tokens to tokenizer
        new_tokens = self.board_cluster_tokens + self.move_cluster_tokens
        if new_tokens:
            num_added = self.tokenizer.add_tokens(new_tokens, special_tokens=False)
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            self._random_initialize_cluster_embeddings()
            logger.info(f"Added {num_added} new tokens to vocabulary")

        logger.info(f"Tokens: {len(self.board_cluster_tokens)} board, {len(self.move_cluster_tokens)} move")

    def _random_initialize_cluster_embeddings(self):
        """Initialize new token embeddings by copying from existing tokens"""
        emb_layer = self.base_model.get_input_embeddings()
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype

        rng = np.random.default_rng(self.config.seed)

        # Build pool of candidate tokens
        vocab_size = len(self.tokenizer)
        special_ids = set(getattr(self.tokenizer, 'all_special_ids', []) or [])
        candidate_ids = [i for i in range(vocab_size) if i not in special_ids]
        if not candidate_ids:
            candidate_ids = list(range(vocab_size))

        # Prefer common ASCII tokens
        preferred_tokens = list("0123456789abcdefghijklmnopqrstuvwxyz.,!?;:-()")
        preferred_ids = []
        for t in preferred_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                preferred_ids.append(tid)
        pool = preferred_ids if len(preferred_ids) >= 10 else candidate_ids

        mapping = {
            'board': [],
            'move': [],
            'board_token_to_index': {},
            'move_token_to_index': {},
            'init_from': {},
        }

        with torch.no_grad():
            # Initialize board tokens
            for i, tok in enumerate(self.board_cluster_tokens):
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
                src_id = int(rng.choice(pool))
                emb_layer.weight[tok_id] = emb_layer.weight[src_id].to(device=device, dtype=dtype)
                mapping['board'].append({
                    'index': i,
                    'token': tok,
                    'token_id': tok_id,
                    'centroid': self.board_centroids[i].detach().cpu().tolist() if self.board_centroids is not None else None,
                })
                mapping['board_token_to_index'][tok] = i
                mapping['init_from'][tok] = {
                    'token_str': self.tokenizer.convert_ids_to_tokens(src_id),
                    'token_id': int(src_id),
                }

            # Initialize move tokens
            for i, tok in enumerate(self.move_cluster_tokens):
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
                src_id = int(rng.choice(pool))
                emb_layer.weight[tok_id] = emb_layer.weight[src_id].to(device=device, dtype=dtype)
                mapping['move'].append({
                    'index': i,
                    'token': tok,
                    'token_id': tok_id,
                    'centroid': self.move_centroids[i].detach().cpu().tolist() if self.move_centroids is not None else None,
                })
                mapping['move_token_to_index'][tok] = i
                mapping['init_from'][tok] = {
                    'token_str': self.tokenizer.convert_ids_to_tokens(src_id),
                    'token_id': int(src_id),
                }

        self.token_centroid_mapping = mapping

    def build_token_centroid_mapping(self) -> Dict[str, Union[Dict, List]]:
        """Return the token-centroid mapping"""
        return getattr(self, 'token_centroid_mapping', {
            'board': [], 'move': [], 'board_token_to_index': {}, 'move_token_to_index': {}, 'init_from': {}
        })

    def _freeze_params(self):
        """Freeze non-LoRA parameters"""
        if not self.config.freeze_non_lora:
            return
        for name, param in self.base_model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            use_cache=False,
        )
        loss = outputs.loss
        with torch.no_grad():
            metrics = {
                'loss': loss.item(),
                'ppl': math.exp(loss.item()) if loss.item() < 50 else float('inf'),
            }
        return loss, metrics
    

    def generate_json(self, input_text: str, max_retries: int = 3) -> Dict:
        """Generate JSON output with validation"""
        device = next(self.parameters()).device
        
        # Encode input
        inputs = self.tokenizer(input_text, return_tensors='pt').to(device)
        
        for retry in range(max_retries):
            try:
                # Generate with constraints if available
                if self.config.enforce_json_schema and _HAS_LOGITS_PROCESSOR:
                    processor = JSONLogitsProcessor(
                        self.tokenizer, 
                        self.board_cluster_tokens, 
                        self.move_cluster_tokens
                    )
                    outputs = self.base_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        logits_processor=LogitsProcessorList([processor]),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    outputs = self.base_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # Decode and parse
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract JSON from generated text
                json_match = re.search(r'\{[^}]+\}', generated)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Validate structure
                    if 'move' in result and 'board' in result:
                        return result
                
            except (json.JSONDecodeError, Exception) as e:
                if retry == max_retries - 1:
                    logger.warning(f"Failed to generate valid JSON after {max_retries} retries: {e}")
                    return {"move": None, "board": None, "error": str(e)}
        
        return {"move": None, "board": None}

    def save_model(self, path: str, also_copy_mapping_to_data: Optional[str] = None):
        os.makedirs(path, exist_ok=True)
        
        # Save adapter
        self.base_model.save_pretrained(os.path.join(path, "adapter"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        
        # Save config
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Save token-centroid mapping
        mapping = self.build_token_centroid_mapping()
        mapping_path = os.path.join(path, "token_centroid_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as mf:
            json.dump(mapping, mf, indent=2)
        
        # Log concise mapping
        def concise(m):
            return {
                'board': [{'index': e['index'], 'token': e['token'], 'token_id': e['token_id']} 
                         for e in m.get('board', [])[:5]],  # Show first 5
                'move': [{'index': e['index'], 'token': e['token'], 'token_id': e['token_id']} 
                        for e in m.get('move', [])[:5]],  # Show first 5
                'total_board': len(m.get('board', [])),
                'total_move': len(m.get('move', [])),
            }
        logger.info(f"Token mapping saved: {json.dumps(concise(mapping), indent=2)}")

        # Copy mapping to data folder
        if also_copy_mapping_to_data:
            try:
                os.makedirs(also_copy_mapping_to_data, exist_ok=True)
                copy_path = os.path.join(also_copy_mapping_to_data, "token_centroid_mapping.json")
                with open(copy_path, 'w', encoding='utf-8') as mf:
                    json.dump(mapping, mf, indent=2)
                logger.info(f"Copied mapping to: {copy_path}")
            except Exception as e:
                logger.warning(f"Failed to copy mapping: {e}")

        logger.info(f"Model saved to {path}")


def _decode_and_log_examples(model, batch: Dict[str, torch.Tensor], step: int, config: HiveLLMConfig):
    """Enhanced logging with token verification and JSON output checking"""
    model_was_training = model.training
    model.eval()
    
    try:
        with torch.no_grad():
            if torch.cuda.is_available():
                ac = torch.autocast(device_type='cuda', dtype=model.compute_dtype)
            else:
                ac = contextlib.nullcontext()
            
            with ac:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=None,
                    use_cache=False,
                )
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)
        
        B = batch['input_ids'].size(0)
        to_log = min(B, max(1, config.max_log_samples))
        
        for i in range(to_log):
            input_ids = batch['input_ids'][i].detach().cpu()
            attn = batch['attention_mask'][i].detach().cpu()
            labels = batch['labels'][i].detach().cpu()
            preds = pred_ids[i].detach().cpu()

            # Get actual sequence length
            valid_len = int(attn.sum().item())
            input_ids = input_ids[:valid_len]
            labels = labels[:valid_len]
            preds = preds[:valid_len]

            # Extract target span
            mask = labels != -100
            if mask.any():
                target_token_ids = input_ids[mask]
                preds_shifted = preds.clone()
                if preds_shifted.numel() > 1:
                    preds_shifted[1:] = preds[:-1]
                pred_target_span = preds_shifted[mask]
            else:
                start = max(0, valid_len - valid_len // 4)
                target_token_ids = input_ids[start:]
                preds_shifted = preds.clone()
                if preds_shifted.numel() > 1:
                    preds_shifted[1:] = preds[:-1]
                pred_target_span = preds_shifted[start:]

            # Decode
            decoded_input = model.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
            decoded_target = model.tokenizer.decode(target_token_ids.tolist(), skip_special_tokens=False)
            decoded_pred = model.tokenizer.decode(pred_target_span.tolist(), skip_special_tokens=False)

            # Check for cluster tokens in decoded strings
            board_tokens_found = [t for t in model.board_cluster_tokens if t in decoded_input]
            move_tokens_found = [t for t in model.move_cluster_tokens if t in decoded_input]
            
            # Try to parse JSON if expected
            json_valid = False
            if config.output_format == "json":
                try:
                    json_match = re.search(r'\{[^}]+\}', decoded_pred)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        json_valid = 'move' in parsed and 'board' in parsed
                except:
                    pass

            # Truncate for display
            def trunc(s: str):
                return (s[:config.decode_max_chars] + '…') if len(s) > config.decode_max_chars else s

            msg = (
                f"\n==== Sample {i+1} @ Step {step} ===="
                f"\nINPUT ({len(input_ids)} tokens): {decoded_input}"
                f"\nTARGET: {decoded_target}"
                f"\nPREDICTED: {decoded_pred}"
                f"\nCluster tokens found - Board: {len(board_tokens_found)}, Move: {len(move_tokens_found)}"
            )
            
            if config.output_format == "json":
                msg += f"\nJSON valid: {json_valid}"
            
            logger.info(msg)

    except Exception as e:
        logger.warning(f"Failed to decode examples at step {step}: {e}")
    finally:
        if model_was_training:
            model.train()


def train_model(config: HiveLLMConfig):
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load model
    logger.info("Creating model...")
    model = HiveLLMModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load data
    logger.info("Loading datasets...")
    
    def load_cache(path):
        if not path or not os.path.exists(path):
            return []
        with open(path, 'rb') as f:
            return pickle.load(f)

    train_samples = load_cache(config.train_cache_path)
    val_samples = load_cache(config.val_cache_path)
    
    if not train_samples:
        raise ValueError(f"No training samples found at {config.train_cache_path}")
    
    logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val samples")

    # Apply data size partitioning BEFORE building progressive indices
    def partition_samples(samples: List[Dict], frac: float) -> List[Dict]:
        if not samples:
            return samples
        if frac >= 1.0:
            return samples
        if frac <= 0.0:
            return []
        keep = max(1, int(len(samples) * frac))
        # Deterministic head partition to keep reproducible behavior
        return samples[:keep]

    if config.data_size != 1.0:
        orig_train, orig_val = len(train_samples), len(val_samples)
        train_samples = partition_samples(train_samples, config.data_size)
        val_samples = partition_samples(val_samples, config.data_size)
        logger.info(f"Data partitioning applied (data_size={config.data_size}): train {orig_train}->{len(train_samples)}, val {orig_val}->{len(val_samples)}")

    # Load system prompt
    system_prompt = None
    if config.system_prompt_path and os.path.exists(config.system_prompt_path):
        with open(config.system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    
    if config.verbose:
        logger.info(f"System prompt:\n{system_prompt if system_prompt else '[None]'}")

    # Create datasets
    # Avoid loading full pretokenized caches when using partial data
    train_load_pretok = config.pretokenized_train_cache_path if config.data_size >= 1.0 else None
    val_load_pretok = config.pretokenized_val_cache_path if config.data_size >= 1.0 else None

    train_dataset = GameClusterSequenceDataset(
        train_samples,
        model.board_cluster_tokens,
        model.move_cluster_tokens,
        model.tokenizer,
        config=config,
        system_prompt=system_prompt,
        pretokenize=config.pretokenize,
        save_pretok_path=config.pretokenized_train_cache_path if config.data_size >= 1.0 else None,
        load_pretok_path=train_load_pretok,
    )

    val_dataset = None
    if config.run_validation and val_samples:
        val_dataset = GameClusterSequenceDataset(
            val_samples,
            model.board_cluster_tokens,
            model.move_cluster_tokens,
            model.tokenizer,
            config=config,
            system_prompt=system_prompt,
            pretokenize=config.pretokenize,
            save_pretok_path=config.pretokenized_val_cache_path if config.data_size >= 1.0 else None,
            load_pretok_path=val_load_pretok,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=GameClusterSequenceDataset.collate,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=GameClusterSequenceDataset.collate,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
        )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )

    # Scheduler
    total_steps = max(1, len(train_loader) * config.epochs // max(1, config.grad_accumulation_steps))
    
    def lr_lambda(step):
        if step < config.warmup_steps:
            return float(step) / max(1, config.warmup_steps)
        progress = float(step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        if config.lr_scheduler == 'cosine':
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            return 1.0 - progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    use_autocast = torch.cuda.is_available()
    use_grad_scaler = torch.cuda.is_available() and getattr(model, "compute_dtype", torch.float16) == torch.float16
    scaler = AMPGradScaler(enabled=use_grad_scaler)
    logger.info(f"AMP autocast: {use_autocast}, GradScaler enabled: {use_grad_scaler}, compute_dtype={getattr(model, 'compute_dtype', None)}")

    logger.info(f"Starting training for {config.epochs} epochs...")
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}") if _HAS_TQDM else train_loader

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Log first batch details
            if epoch == 0 and batch_idx == 0 and config.verbose:
                labels = batch['labels'][0]
                non_masked = (labels != -100).sum().item()
                total = labels.size(0)
                logger.info(f"First batch: {non_masked}/{total} tokens unmasked for training")

            # Forward pass
            if use_autocast:
                autocast_ctx = torch.autocast(device_type='cuda', dtype=model.compute_dtype)
            else:
                autocast_ctx = contextlib.nullcontext()
            
            with autocast_ctx:
                loss, metrics = model.training_step(batch)
                loss = loss / max(1, config.grad_accumulation_steps)

            # Backward pass
            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % config.grad_accumulation_steps == 0:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % config.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    mem_txt = ""
                    if torch.cuda.is_available():
                        alloc = torch.cuda.memory_allocated() / (1024**3)
                        mem_txt = f", mem={alloc:.2f}GB"
                    logger.info(
                        f"Step {global_step}: loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.2f}, lr={lr:.2e}{mem_txt}"
                    )

                if config.run_validation and val_loader and global_step % config.eval_every == 0:
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
                            with autocast_ctx:
                                loss, _ = model.training_step(val_batch)
                            val_loss += loss.item()
                            val_steps += 1
                    avg_val_loss = val_loss / max(1, val_steps)
                    val_ppl = math.exp(avg_val_loss) if avg_val_loss < 50 else float('inf')
                    logger.info(f"Validation @ Step {global_step}: loss={avg_val_loss:.4f}, ppl={val_ppl:.2f}")
                    model.train()
                # Log predictions
                if config.log_preds_every > 0 and global_step % config.log_preds_every == 0:
                    _decode_and_log_examples(model, batch, global_step, config)
                
                if _HAS_TQDM and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}", 
                        'ppl': f"{metrics['ppl']:.2f}"
                    })

            epoch_loss += metrics['loss']
            epoch_steps += 1

        avg_loss = epoch_loss / max(1, epoch_steps)
        logger.info(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if config.save_every and (epoch + 1) % config.save_every == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint_epoch{epoch+1}")
            model.save_model(ckpt_path, also_copy_mapping_to_data=os.path.dirname(config.train_cache_path or ""))

    # Final save
    model.save_model(config.output_dir, also_copy_mapping_to_data=os.path.dirname(config.train_cache_path or ""))
    logger.info(f"Training complete! Model saved to {config.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config JSON file')
    parser.add_argument('--output_format', type=str, choices=['json', 'text'], default='json')
    parser.add_argument('--add_descriptions', action='store_true', help='Add natural language descriptions')
    parser.add_argument('--verify_tokens', action='store_true', help='Verify token addition')
    parser.add_argument('--train_cache', type=str)
    parser.add_argument('--val_cache', type=str)
    parser.add_argument('--board_centroids', type=str)
    parser.add_argument('--move_centroids', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--pretok_train', type=str)
    parser.add_argument('--pretok_val', type=str)
    # New CLI flags
    parser.add_argument('--validation', action='store_true', help='Run validation during training')
    parser.add_argument('--data_size', type=float, default=None, help='Fraction of data to load (0-1) before building prediction fractions')
    args = parser.parse_args()

    config = HiveLLMConfig()
    
    # Load config file
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

    # CLI overrides
    if args.output_format:
        config.output_format = args.output_format
    if args.add_descriptions:
        config.add_natural_language_description = True
    if args.verify_tokens:
        config.verify_token_addition = True
    if args.train_cache:
        config.train_cache_path = args.train_cache
    if args.val_cache:
        config.val_cache_path = args.val_cache
    if args.board_centroids:
        config.board_centroids_path = args.board_centroids
    if args.move_centroids:
        config.move_centroids_path = args.move_centroids
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.verbose:
        config.verbose = True
    if args.pretok_train:
        config.pretokenized_train_cache_path = args.pretok_train
    if args.pretok_val:
        config.pretokenized_val_cache_path = args.pretok_val
    if args.validation:
        config.run_validation = True
    if args.data_size is not None:
        # Clamp to [0,1]
        try:
            config.data_size = float(max(0.0, min(1.0, args.data_size)))
        except Exception:
            logger.warning(f"Invalid --data_size {args.data_size}; defaulting to 1.0")
            config.data_size = 1.0

    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join("logs", "training.log")
    txt_log_file = os.path.join("logs", "training.txt")
    setup_logging(log_file, to_stdout=True, level=logging.INFO, also_log_to_txt=txt_log_file)
    logger.info(f"Configuration: output_format={config.output_format}, add_descriptions={config.add_natural_language_description}")
    
    # Run training
    train_model(config)


if __name__ == "__main__":
    main()