#!/usr/bin/env python3
"""
HIVE LLM architecture with GPT-OSS via Unsloth - VERBOSE + PRETOKENIZATION + RANDOM-INIT MAPPING
- Adds rich verbose prints (sequence composition, system prompt at start)
- Pretokenizes with tqdm and caches to disk for fast reloads
- Logs GPU usage and device offloading
- Reworks cluster token initialization: NO linear projection. Instead, randomly map
  each new cluster token's embedding to a copy of an existing vocab token's embedding
  (dictionary-based mapping), and save that mapping.
- Saves a token↔centroid mapping JSON (plus init-from token) into the model output dir,
  and also copies it into the dataset folder for convenience.
- Saves pretokenized datasets (input_ids/attention_mask/labels) to data folders and supports reload.
- Updates autocast to the new torch.autocast API. Uses torch.amp.GradScaler when available.

Note: This file supersedes the earlier script and fixes indentation, AMP usage,
      and training no_grad bug in forward.
"""
import os
import json
import math
import pickle
import logging
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

# AMP (prefer new torch.amp.GradScaler; fallback to old path)
try:
    from torch.amp import GradScaler as AMPGradScaler  # PyTorch ≥2.0
except Exception:
    from torch.cuda.amp import GradScaler as AMPGradScaler
# Windows torch.compile fix
#if os.name == 'nt':
#    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


logger = logging.getLogger("HiveLLM")

def setup_logging(log_path: str, to_stdout: bool = True, level=logging.INFO, also_log_to_txt: Optional[str] = None):
    # Nuke any existing handlers configured by other libs / notebooks
    logging.basicConfig(force=True)
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Optional second file (e.g., .txt) to satisfy dual logging request
    if also_log_to_txt is not None:
        try:
            ftxt = logging.FileHandler(also_log_to_txt, mode='a', encoding='utf-8')
            ftxt.setFormatter(fmt)
            ftxt.setLevel(level)
            logger.addHandler(ftxt)
        except Exception as e:
            # Don't break logging if secondary handler fails
            logger.warning(f"Failed to attach secondary log file handler ({also_log_to_txt}): {e}")

    if to_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    # Avoid double-printing up the root chain
    logger.propagate = False

@dataclass
class HiveLLMConfig:
    """Configuration for HIVE LLM"""
    # Model
    base_model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"

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
    learning_rate: float = 5e-5
    batch_size: int = 2
    epochs: int = 1
    grad_accumulation_steps: int = 32
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 1
    eval_every: int = 200
    save_every: int = 0
    lr_scheduler: str = "cosine"
    seed: int = 42

    # Logging of IO/predictions
    log_preds_every: int = 5  # how often (in optimizer steps) to log input/target/prediction triplets
    max_log_samples: int = 4    # samples per logging event
    decode_max_chars: int = 500 # truncate decoded strings for readability

    # Paths
    output_dir: str = "models/hive_llm"
    gnn_model_path: str = "src/models/pretrain_GIN_3.pt"
    board_centroids_path: Optional[str] = "clustering_models/boards/cluster_centroids_kmeans_best.pkl"
    move_centroids_path: Optional[str] = "clustering_models/moves/cluster_centroids_kmeans_best.pkl"
    train_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/train_sequential_cache.pkl"
    val_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/validation_sequential_cache.pkl"

    # New: pretokenized caches to speed reloads
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
    system_prompt_path: Optional[str] = None
    user_prompt_prefix: str = "Game cluster sequence:"
    verbose: bool = False
    use_cache: bool = False
    enable_gradient_checkpointing: bool = True
    freeze_non_lora: bool = True

    # Data/tokenization
    pretokenize: bool = False
    pretokenize_show_samples: int = 3  # how many sequence-composition examples to log


class GameClusterSequenceDataset(Dataset):
    """Dataset for cluster token sequences with label masking.
    Supports optional on-disk pretokenization caches for speed.
    """

    def __init__(
        self,
        samples: List[Dict],
        board_tokens: List[str],
        move_tokens: List[str],
        tokenizer,
        add_eos: bool = True,
        system_prompt: Optional[str] = None,
        user_prefix: str = "",
        responses_only: bool = True,
        pretokenize: bool = False,
        verbose: bool = False,
        save_pretok_path: Optional[str] = None,
        load_pretok_path: Optional[str] = None,
    ):
        self.samples = samples
        self.board_tokens = board_tokens
        self.move_tokens = move_tokens
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.system_prompt = system_prompt
        self.user_prefix = user_prefix
        self.responses_only = responses_only
        self.num_board = len(board_tokens)
        self.num_move = len(move_tokens)
        self.verbose = verbose
        self._encoded: Optional[List[Dict[str, torch.Tensor]]] = None

        # Try to load pretokenized cache if requested and present
        if load_pretok_path and os.path.exists(load_pretok_path):
            logger.info(f"Loading pretokenized dataset from: {load_pretok_path}")
            with open(load_pretok_path, 'rb') as f:
                raw = pickle.load(f)
            # Convert lists back to tensors (with progress)
            self._encoded = []
            iterator = tqdm(range(len(raw)), desc="Loading pretok", leave=False) if _HAS_TQDM else range(len(raw))
            for i in iterator:
                item = raw[i]
                self._encoded.append({
                    'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
                    'labels': torch.tensor(item['labels'], dtype=torch.long),
                })
            logger.info(f"Pretokenized samples: {len(self._encoded)}")
            return

        # Optional pre-tokenization for speed (and save to disk)
        if pretokenize:
            if _HAS_TQDM:
                logger.info("Pretokenizing dataset with tqdm for faster training…")
            self._encoded = []
            iterator = tqdm(range(len(self.samples)), desc="Pretokenizing", leave=False) if _HAS_TQDM else range(len(self.samples))
            shown = 0
            for idx in iterator:
                item = self._encode_item(idx)
                self._encoded.append({
                    'input_ids': item['input_ids'],
                    'attention_mask': item['attention_mask'],
                    'labels': item['labels'],
                })
                # Show sequence composition examples
                if self.verbose and shown < 3:
                    shown += 1
                    chat = item.get('chat', [])
                    assistant = item.get('assistant_text', '')
                    context_user = next((m['content'] for m in chat if m.get('role') == 'user'), '')
                    logger.info(
                        f"[Sample {shown}] Sequence composition:\n  User/context: {context_user}\n  Assistant/target: {assistant}\n  input_ids={len(item['input_ids'])}, labels_unmasked={(item['labels'] != -100).sum().item()}"
                    )
            # Save pretokenized cache if a path was provided
            if save_pretok_path:
                try:
                    out_dir = os.path.dirname(save_pretok_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                    # Store as lists for portability
                    dumpable = [
                        {
                            'input_ids': t['input_ids'].tolist(),
                            'attention_mask': t['attention_mask'].tolist(),
                            'labels': t['labels'].tolist(),
                        } for t in self._encoded
                    ]
                    with open(save_pretok_path, 'wb') as f:
                        pickle.dump(dumpable, f)
                    logger.info(f"Saved pretokenized dataset to: {save_pretok_path}")
                except Exception as e:
                    logger.warning(f"Failed to save pretokenized dataset to {save_pretok_path}: {e}")

    def __len__(self):
        return len(self.samples) if self._encoded is None else len(self._encoded)

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

    def _build_next_pair(self, b_seq: List[int], m_seq: List[int]) -> Tuple[str, str]:
        """Build context and target for next-pair prediction."""
        if len(b_seq) < 2 or len(m_seq) < 1:
            return "", ""

        pairs = min(len(m_seq), len(b_seq) - 1)
        if pairs < 1:
            return "", ""

        target_move_idx = pairs - 1

        # Build context (everything before target pair)
        context_tokens = []
        for i in range(target_move_idx):
            if 0 <= b_seq[i] < self.num_board:
                context_tokens.append(self.board_tokens[b_seq[i]])
            if 0 <= m_seq[i] < self.num_move:
                context_tokens.append(self.move_tokens[m_seq[i]])

        # Add board before target move
        if 0 <= b_seq[target_move_idx] < self.num_board:
            context_tokens.append(self.board_tokens[b_seq[target_move_idx]])

        # Build target (move + board after)
        target_tokens = []
        if 0 <= m_seq[target_move_idx] < self.num_move:
            target_tokens.append(self.move_tokens[m_seq[target_move_idx]])
        if target_move_idx + 1 < len(b_seq) and 0 <= b_seq[target_move_idx + 1] < self.num_board:
            target_tokens.append(self.board_tokens[b_seq[target_move_idx + 1]])

        context = " ".join(context_tokens)
        target = " ".join(target_tokens)

        logger.info(f"Built context: '{context}'")
        logger.info(f"Built target: '{target}'")

        return context, target

    def __getitem__(self, idx):
        if self._encoded is not None:
            return self._encoded[idx]
        return self._encode_item(idx)

    def _encode_item(self, idx):
        sample = self.samples[idx]
        b_seq = self._norm_seq(sample.get('board_cluster_ids_sequence'))
        m_seq = self._norm_seq(sample.get('chosen_move_cluster_ids_sequence'))

        # Build next-pair prediction
        context, target = self._build_next_pair(b_seq, m_seq)

        # Build chat
        chat = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})

        user_msg = context
        if self.user_prefix:
            user_msg = f"{self.user_prefix} {user_msg}".strip()

        chat.append({"role": "user", "content": user_msg})
        chat.append({"role": "assistant", "content": target})

        # Apply chat template
        try:
            rendered = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback rendering
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

        # Proper label masking: train only on assistant response tokens
        labels = input_ids.clone()
        if self.responses_only and target:
            assistant_tokens = self.tokenizer(
                target,
                add_special_tokens=False,
                return_tensors='pt',
            )['input_ids'][0]
            assistant_len = len(assistant_tokens)
            if assistant_len > 0:
                found = False
                # Search from end
                for start_idx in range(len(input_ids) - assistant_len, -1, -1):
                    if torch.equal(input_ids[start_idx:start_idx + assistant_len], assistant_tokens):
                        labels[:start_idx] = -100
                        found = True
                        break
                if not found:
                    logger.warning(f"Could not find exact assistant response in tokenized output for sample {idx}")
                    cutoff = int(len(labels) * 0.7)
                    labels[:cutoff] = -100
            else:
                logger.warning(f"Empty assistant response for sample {idx}")
                labels[:] = -100
        elif not self.responses_only:
            pass  # train on full sequence
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
        """Collate batch with right-padding and label masking on pads."""
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
    """HIVE LLM model with GPT-OSS backbone.
    - Random dictionary-based init for new cluster tokens (no linear projection).
    - Saves detailed token-centroid and init-from mappings.
    """

    def __init__(self, config: HiveLLMConfig):
        super().__init__()
        self.config = config
        logger.info(f"Initializing HiveLLM with base model: {config.base_model_name}")
        self._load_backbone()
        self._load_centroids_and_add_tokens()
        self._freeze_params()
        self._log_device_info()

    def _load_backbone(self):
        name = self.config.base_model_name
        use_unsloth = name.startswith("unsloth/") and _HAS_UNSLOTH

        # Choose computation dtype
        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if use_unsloth:
            logger.info("Loading backbone via Unsloth FastLanguageModel …")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=name,
                max_seq_length=self.config.max_seq_length,
                dtype=self.compute_dtype,
                load_in_4bit=self.config.use_4bit,
            )
            # Apply LoRA
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
        else:
            logger.warning("Unsloth not available or model is not unsloth/* — falling back to HF loading")
            self._load_standard_hf()

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_standard_hf(self):
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

    def _log_device_info(self):
        """Log GPU/offloading and memory info."""
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
                logger.info(f"CUDA memory at init — allocated {alloc:.2f} GB, reserved {reserved:.2f} GB")
            dm = getattr(self.base_model, 'hf_device_map', None)
            if dm:
                cpu_layers = [k for k, v in dm.items() if isinstance(v, str) and v.startswith('cpu')]
                disk_layers = [k for k, v in dm.items() if isinstance(v, str) and v.startswith('disk')]
                logger.info(f"Device map entries: {len(dm)}; CPU offload: {len(cpu_layers)}, disk offload: {len(disk_layers)}")
            else:
                logger.info("No explicit device map found; model likely on a single device or managed by backend.")
        except Exception as e:
            logger.warning(f"Could not log device info: {e}")

    def _load_centroids_and_add_tokens(self):
        """Load cluster centroids and add corresponding tokens to tokenizer.
        Initialize new token embeddings by copying from random existing tokens (dictionary mapping).
        """
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
            self.tokenizer.add_tokens(new_tokens, special_tokens=False)
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            self._random_initialize_cluster_embeddings()

        logger.info(f"Added {len(self.board_cluster_tokens)} board tokens, {len(self.move_cluster_tokens)} move tokens")

    def _random_initialize_cluster_embeddings(self):
        """Instead of projecting centroids with a linear layer, randomly copy embeddings
        from a chosen pool of existing vocab tokens. Save the mapping for reproducibility.
        """
        emb_layer = self.base_model.get_input_embeddings()
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype

        rng = np.random.default_rng(self.config.seed)

        # Build a pool of candidate token IDs to copy from (avoid specials)
        vocab_size = len(self.tokenizer)
        special_ids = set(getattr(self.tokenizer, 'all_special_ids', []) or [])
        candidate_ids = [i for i in range(vocab_size) if i not in special_ids]
        if not candidate_ids:
            candidate_ids = list(range(vocab_size))

        # Prefer some common ASCII tokens when available
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
            'init_from': {},  # token -> {token_str, token_id}
        }

        with torch.no_grad():
            # Board tokens
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

            # Move tokens
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

        # Keep mapping for saving later
        self.token_centroid_mapping: Dict[str, Union[Dict, List]] = mapping

    def build_token_centroid_mapping(self) -> Dict[str, Union[Dict, List]]:
        """Expose the prepared mapping (includes centroid vectors and init-from token)."""
        return getattr(self, 'token_centroid_mapping', {
            'board': [], 'move': [], 'board_token_to_index': {}, 'move_token_to_index': {}, 'init_from': {}
        })

    def _freeze_params(self):
        """Freeze non-LoRA parameters."""
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
        # Save token-centroid/init mapping
        mapping = self.build_token_centroid_mapping()
        mapping_path = os.path.join(path, "token_centroid_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as mf:
            json.dump(mapping, mf, indent=2)
        # Print a concise mapping to logs (without vectors)
        def concise(m):
            return {
                'board': [{'index': e['index'], 'token': e['token'], 'token_id': e['token_id']} for e in m.get('board', [])],
                'move': [{'index': e['index'], 'token': e['token'], 'token_id': e['token_id']} for e in m.get('move', [])],
                'init_from': {k: v for k, v in m.get('init_from', {}).items()},
            }
        logger.info(f"Token-centroid mapping saved to {mapping_path}\n{json.dumps(concise(mapping), indent=2)}")

        # Optionally copy mapping into a data folder for convenience
        if also_copy_mapping_to_data:
            try:
                os.makedirs(also_copy_mapping_to_data, exist_ok=True)
                copy_path = os.path.join(also_copy_mapping_to_data, "token_centroid_mapping.json")
                with open(copy_path, 'w', encoding='utf-8') as mf:
                    json.dump(mapping, mf, indent=2)
                logger.info(f"Also copied mapping to data dir: {copy_path}")
            except Exception as e:
                logger.warning(f"Failed to copy mapping to data dir: {e}")

        logger.info(f"Model artifacts saved to {path}")


def train_model(config: HiveLLMConfig):
    # Seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load model
    logger.info("Creating model…")
    model = HiveLLMModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load data caches (raw samples)
    logger.info("Loading datasets…")

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

    # Load system prompt and print once at start when verbose
    system_prompt = None
    if config.system_prompt_path and os.path.exists(config.system_prompt_path):
        with open(config.system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    if config.verbose:
        logger.info("System prompt (at training start):\n" + (system_prompt if system_prompt else "[None]"))

    # Create datasets (support pretokenized caches on disk)
    train_dataset = GameClusterSequenceDataset(
        train_samples,
        model.board_cluster_tokens,
        model.move_cluster_tokens,
        model.tokenizer,
        add_eos=config.add_eos_token,
        system_prompt=system_prompt,
        user_prefix=config.user_prompt_prefix,
        responses_only=True,
        pretokenize=config.pretokenize,
        verbose=config.verbose,
        save_pretok_path=config.pretokenized_train_cache_path,
        load_pretok_path=config.pretokenized_train_cache_path,
    )

    val_dataset = None
    val_loader = None
    if val_samples:
        val_dataset = GameClusterSequenceDataset(
            val_samples,
            model.board_cluster_tokens,
            model.move_cluster_tokens,
            model.tokenizer,
            add_eos=config.add_eos_token,
            system_prompt=system_prompt,
            user_prefix=config.user_prompt_prefix,
            responses_only=True,
            pretokenize=config.pretokenize,
            verbose=config.verbose,
            save_pretok_path=config.pretokenized_val_cache_path,
            load_pretok_path=config.pretokenized_val_cache_path,
        )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=GameClusterSequenceDataset.collate,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
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
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

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
    use_amp = torch.cuda.is_available()
    scaler = AMPGradScaler(enabled=use_amp)

    logger.info(f"Starting training for {config.epochs} epochs…")
    global_step = 0

    def _decode_and_log_examples(batch: Dict[str, torch.Tensor], step: int):
        """Decode and log input/target/prediction triplets for a few samples.
        Uses positions where labels != -100 as the target span.
        """
        model_was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                # NEW: mirror training autocast
                if torch.cuda.is_available():
                    ac = torch.autocast(device_type='cuda', dtype=model.compute_dtype)
                else:
                    import contextlib
                    ac = contextlib.nullcontext()
                with ac:
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=None,
                        use_cache=False,
                    )
                logits = outputs.logits
                pred_ids_all = torch.argmax(logits, dim=-1)
            B = batch['input_ids'].size(0)
            to_log = min(B, max(1, config.max_log_samples))
            for i in range(to_log):
                input_ids = batch['input_ids'][i].detach().cpu()
                attn = batch['attention_mask'][i].detach().cpu()
                labels = batch['labels'][i].detach().cpu()
                preds = pred_ids_all[i].detach().cpu()

                # Trim to actual length via attention mask
                valid_len = int(attn.sum().item()) if attn.ndim == 1 else input_ids.size(0)
                input_ids = input_ids[:valid_len]
                labels = labels[:valid_len]
                preds = preds[:valid_len]

                # Extract target span tokens (where labels != -100)
                mask = labels != -100
                if mask.any():
                    target_token_ids = input_ids[mask]
                    # Align predictions: label at position t corresponds to logits at position t-1
                    preds_shifted = preds.clone()
                    if preds_shifted.numel() > 1:
                        preds_shifted[1:] = preds[:-1]
                    pred_token_ids = preds_shifted[mask]
                else:
                    # Fallback: use last quarter of the sequence
                    start = max(0, valid_len - valid_len // 4)
                    target_token_ids = input_ids[start:]
                    preds_shifted = preds.clone()
                    if preds_shifted.numel() > 1:
                        preds_shifted[1:] = preds[:-1]
                    pred_token_ids = preds_shifted[start:]

                # Decode
                decoded_input = model.tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
                decoded_target = model.tokenizer.decode(target_token_ids.tolist(), skip_special_tokens=True)
                decoded_pred = model.tokenizer.decode(pred_token_ids.tolist(), skip_special_tokens=True)

                # Truncate for readability
                def trunc(s: str):
                    return (s[: config.decode_max_chars] + '…') if len(s) > config.decode_max_chars else s

                msg = (
                    f"\n==== IO @ step {step}, sample {i} ===="
                    f"\nINPUT:\n{trunc(decoded_input)}"
                    f"\nTARGET:\n{trunc(decoded_target)}"
                    f"\nPRED:\n{trunc(decoded_pred)}\n=============================="
                )
                logger.info(msg)

        except Exception as e:
            logger.warning(f"Failed to decode/log examples at step {step}: {e}")
        finally:
            if model_was_training:
                model.train()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}") if _HAS_TQDM else train_loader

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Debug first batch labels
            if epoch == 0 and batch_idx == 0 and config.verbose:
                labels = batch['labels'][0]
                non_masked = (labels != -100).sum().item()
                total = labels.size(0)
                logger.info(f"First batch: {non_masked}/{total} tokens unmasked for training")
                unmasked_ids = labels[labels != -100]
                if len(unmasked_ids) > 0:
                    text = model.tokenizer.decode(unmasked_ids[:])
                    logger.info(f"Training on: {text}…")

            # Forward with new autocast API (no no_grad!)
            if use_amp:
                autocast_ctx = torch.autocast(device_type='cuda', dtype=model.compute_dtype)
            else:
                autocast_ctx = contextlib.nullcontext()
            with autocast_ctx:
                loss, metrics = model.training_step(batch)
                loss = loss / max(1, config.grad_accumulation_steps)

            # Backward + step
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % config.grad_accumulation_steps == 0:
                if use_amp:
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
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        mem_txt = f", cuda_mem_alloc={alloc:.2f}GB, reserved={reserved:.2f}GB"
                    logger.info(
                        f"Step {global_step}: loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.2f}, lr={lr:.2e}{mem_txt}"
                    )
                # Log input/target/predictions occasionally
                if config.log_preds_every > 0 and global_step % config.log_preds_every == 0:
                    # Use the same batch for a quick preview
                    _decode_and_log_examples(batch, global_step)
                if _HAS_TQDM and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'ppl': f"{metrics['ppl']:.2f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

            epoch_loss += metrics['loss']
            epoch_steps += 1

        avg_loss = epoch_loss / max(1, epoch_steps)
        logger.info(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

        # Quick validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for b in val_loader:
                    b = {k: v.to(device) for k, v in b.items()}
                    loss, metrics = model.training_step(b)
                    val_loss += metrics['loss']
                    val_steps += 1
                    if val_steps >= 10:
                        break
            avg_val_loss = val_loss / max(1, val_steps)
            val_ppl = math.exp(avg_val_loss) if avg_val_loss < 50 else float('inf')
            logger.info(f"Validation: loss={avg_val_loss:.4f}, ppl={val_ppl:.2f}")

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
    parser.add_argument('--train_cache', type=str)
    parser.add_argument('--val_cache', type=str)
    parser.add_argument('--board_centroids', type=str)
    parser.add_argument('--move_centroids', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--pretok_train', type=str, help='Path to pretokenized train cache (load/save)')
    parser.add_argument('--pretok_val', type=str, help='Path to pretokenized val cache (load/save)')
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

    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, "training.log")
    txt_log_file = os.path.join(config.output_dir, "training.txt")
    setup_logging(log_file, to_stdout=True, level=logging.INFO, also_log_to_txt=txt_log_file)
    logger.info(f"Logging initialized. Writing to: {log_file}")
    # Run training
    train_model(config)


if __name__ == "__main__":
    main()
