#!/usr/bin/env python3
"""
Modified HIVE LLM trainer with sequential context and move similarity module.
Implements the new prompt format with historical board states and moves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from tqdm import tqdm
from torch import bfloat16

# Hugging Face and LoRA imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)

# Local imports
from improved_projection_layers import HiveProjectionModule, ProjectionConfig
from llm_answer_extractor import HiveLLMAnswerExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SequentialHiveLLMConfig:
    """Configuration for sequential HIVE LLM training"""
    
    # Model configuration
    base_model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    gin_embedding_dim: int = 256
    llm_token_dim: int = 4096
    move_embedding_mode: str = "difference"
    sequence_length: int = 6  # Number of historical (board, move) pairs
    
    # Projection layer configuration
    projection_hidden_dim: int = 512
    projection_intermediate_dim: int = 1024
    projection_dropout: float = 0.1
    projection_num_layers: int = 3
    projection_activation: str = "gelu"
    use_layer_norm: bool = True
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization configuration
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Training configuration
    max_seq_length: int = 2048
    learning_rate: float = 5e-5
    projection_lr: float = 2e-4
    num_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    use_mixed_precision: bool = True
    nan_tolerance: int = 10
    use_cache: bool = False
    # Free-generation and logging
    use_free_generation: bool = False  # If True, let the model generate text and extract tokens instead of using placeholders
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    print_every_n_steps: int = 1  # Print model output every N optimizer steps (optimizer steps, not batches)
    
    # Loss configuration
    move_loss_weight: float = 1.0
    state_loss_weight: float = 0.5
    classification_loss_weight: float = 2.0
    use_hybrid_loss: bool = True
    
    # Special token IDs (will be set during initialization)
    board_token_id: Optional[int] = None
    move_token_id: Optional[int] = None
    chosen_move_token_id: Optional[int] = None
    next_state_token_id: Optional[int] = None
    
    # Paths
    output_dir: str = "models/LLM_1000_tournament_mse"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Pretrained autoencoder paths (encoder: GIN->LLM, decoder: LLM->GIN)
    ae_dir: str = "ae_4096_tournament"
    ae_encoder_path: Optional[str] = None  # defaults to f"{ae_dir}/encoder.pt"
    ae_decoder_path: Optional[str] = None  # defaults to f"{ae_dir}/decoder.pt"


class AutoencoderProjectionBridge(nn.Module):
    """Frozen bridge using pretrained encoder/decoder between GIN and LLM spaces.

    Exposes the same minimal API used elsewhere:
    - state_projection: [B, 1, gin_dim] or [B, gin_dim] -> [B, llm_dim]
    - move_projection.forward(moves[, mask]): [B, M, 1, gin_dim] or [B, M, gin_dim] -> [B, M, llm_dim]
    - move_projection.forward_single: [B, 1, gin_dim] or [B, gin_dim] -> [B, llm_dim]
    - state_projection_inverse: [B, llm_dim] -> [B, gin_dim]
    - move_projection_inverse: [B, llm_dim] -> [B, gin_dim]
    """

    class _MoveAE(nn.Module):
        def __init__(self, encoder: nn.Module):
            super().__init__()
            self.encoder = encoder

        def forward(self, moves: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # moves: [B, M, 1, D] or [B, M, D]
            if moves.dim() == 4:
                B, M, _, D = moves.shape
                x = moves.view(B, M, D)
            elif moves.dim() == 3:
                B, M, D = moves.shape
                x = moves
            else:
                raise ValueError(f"Unexpected move tensor shape: {moves.shape}")
            y = self.encoder(x.reshape(-1, x.shape[-1]))  # [B*M, llm]
            return y.view(B, M, -1)

        def forward_single(self, move: torch.Tensor) -> torch.Tensor:
            # move: [B, 1, D] or [B, D]
            if move.dim() == 3:
                B, _, D = move.shape
                x = move.view(B, D)
            elif move.dim() == 2:
                x = move
            else:
                raise ValueError(f"Unexpected move tensor shape: {move.shape}")
            return self.encoder(x)

    def __init__(self, encoder_path: str, decoder_path: str, gin_dim: int, llm_dim: int, device: Optional[torch.device] = None):
        super().__init__()
        # Build modules matching saved state_dicts
        self.encoder, self.decoder = self._build_modules(encoder_path, decoder_path, gin_dim, llm_dim)

        # Freeze and eval by default
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()

        # Wrap move projection API
        self.move_projection = AutoencoderProjectionBridge._MoveAE(self.encoder)

        if device is not None:
            self.to(device)

    def _build_modules(self, enc_path: str, dec_path: str, gin_dim: int, llm_dim: int) -> Tuple[nn.Module, nn.Module]:
        # Try to import the autoencoder components if available
        EncoderCls = None
        DecoderCls = None
        try:
            from ai.autoencoder import Encoder as AEEncoder, Decoder as AEDecoder
            EncoderCls = AEEncoder
            DecoderCls = AEDecoder
        except Exception:
            pass

        if EncoderCls is None or DecoderCls is None:
            # Fallback simple MLPs if classes not importable
            def simple_mlp(in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, max(in_dim // 2, 1)), nn.ReLU(),
                    nn.Linear(max(in_dim // 2, 1), max(in_dim // 4, 1)), nn.ReLU(),
                    nn.Linear(max(in_dim // 4, 1), out_dim)
                )
            encoder = simple_mlp(gin_dim, llm_dim)
            decoder = simple_mlp(llm_dim, gin_dim)
        else:
            encoder = EncoderCls(gin_dim, llm_dim)
            decoder = DecoderCls(llm_dim, gin_dim)

        # Load state dicts
        try:
            enc_state = torch.load(enc_path, map_location='cpu')
            if isinstance(enc_state, dict):
                encoder.load_state_dict(enc_state, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load encoder state_dict from {enc_path}: {e}")

        try:
            dec_state = torch.load(dec_path, map_location='cpu')
            if isinstance(dec_state, dict):
                decoder.load_state_dict(dec_state, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load decoder state_dict from {dec_path}: {e}")

        return encoder, decoder

    # GIN -> LLM
    def state_projection(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.encoder(x)

    # LLM -> GIN
    def state_projection_inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.squeeze(1)
        return self.decoder(y)

    # Inverse for moves: used by extractor; expects [B, llm_dim] -> [B, gin_dim]
    def move_projection_inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.squeeze(1)
        return self.decoder(y)


class SequentialHiveDataset(Dataset):
    """Dataset for sequential HIVE LLM training"""
    
    def __init__(self, data_dir: str, split: str = "train", config: SequentialHiveLLMConfig = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config
        
        # Load sequential dataset
        self.load_dataset()
        
    def load_dataset(self):
        """Load the sequential HIVE dataset"""
        logger.info(f"Loading {self.split} sequential dataset from {self.data_dir}")
        
        # Load from cache file (from the sequential data generation script)
        cache_file = self.data_dir / f"{self.split}_sequential_cache.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
            logger.info(f"Loaded {len(self.samples)} sequential samples from cache")
        else:
            # Try loading from individual files if cache doesn't exist
            raise FileNotFoundError(f"Sequential cache file not found: {cache_file}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'context_board_embeddings': sample['context_board_embeddings'],
            'context_move_embeddings': sample['context_move_embeddings'],
            'current_board_embedding': sample['current_board_embedding'],
            'legal_move_embeddings': sample['legal_move_embeddings'],
            'legal_move_texts': sample['legal_move_texts'],
            'chosen_move_idx': sample['chosen_move_idx'],
            'chosen_move_embedding': sample['chosen_move_embedding'],
            'next_board_embedding': sample['next_board_embedding'],
            'sequence_length': sample['sequence_length']
        }


class MoveSimilarityModule(nn.Module):
    """Non-trainable module for finding closest legal move during inference"""
    
    def __init__(self, config: SequentialHiveLLMConfig):
        super().__init__()
        self.config = config
        
    def find_closest_move(self, 
                         predicted_move: torch.Tensor,
                         legal_moves: torch.Tensor,
                         legal_move_mask: torch.Tensor,
                         projection_module: nn.Module) -> Tuple[int, float]:
        """
        Find the legal move closest to the predicted move (all in LLM space)
        
        Args:
            predicted_move: [llm_token_dim] - LLM predicted move embedding
            legal_moves: [num_legal_moves, 1, gin_dim] or [num_legal_moves, gin_dim] - Legal move embeddings in GIN space
            legal_move_mask: [num_legal_moves] - Mask for valid moves
            projection_module: Module that contains the encoder to project GIN->LLM
            
        Returns:
            best_move_idx: Index of closest legal move
            score: -MSE score in LLM space (higher is better)
        """
        with torch.no_grad():
            # Ensure predicted vector is [D]
            if predicted_move.dim() == 2:
                pred_llm = predicted_move.squeeze(0)
            elif predicted_move.dim() == 1:
                pred_llm = predicted_move
            else:
                pred_llm = predicted_move.reshape(-1, predicted_move.shape[-1])[-1]

            # Project legal moves from GIN -> LLM using the frozen encoder
            # Accept [N,1,Dgin] or [N,Dgin]
            if legal_moves.dim() == 3:
                # add batch dim -> [1, N, 1, Dgin]
                lm_llm = projection_module.move_projection(legal_moves.unsqueeze(0)).squeeze(0)  # [N, Dllm]
            elif legal_moves.dim() == 2:
                lm_llm = projection_module.move_projection(legal_moves.unsqueeze(0)).squeeze(0)  # [N, Dllm]
            else:
                raise ValueError(f"Unexpected legal_moves shape: {legal_moves.shape}")

            # MSE distance in LLM space -> use negative MSE as score so higher is better
            diffs = lm_llm - pred_llm.unsqueeze(0)      # [N, D]
            mse = (diffs.pow(2).mean(dim=1))            # [N]
            scores = -mse                                # higher is better

            # Mask out invalid moves
            masked_scores = scores.masked_fill(~legal_move_mask, -float('inf'))

            # Find best move (max score == min MSE)
            best_move_idx = masked_scores.argmax().item()
            best_score = masked_scores[best_move_idx].item()

            return best_move_idx, best_score


class HybridDualLoss(nn.Module):
    """Stable MSE loss for move and state (LLM space)."""

    def __init__(self, move_weight: float = 1.0, state_weight: float = 1.0, 
                 classification_weight: float = 0.0, use_hybrid: bool = False):
        super().__init__()
        self.move_weight = move_weight
        self.state_weight = state_weight

    def forward(
        self,
        predicted_move: torch.Tensor,
        target_move: torch.Tensor,
        predicted_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute mean squared error (MSE) loss for move and state predictions."""

        # Clamp to prevent exploding values while preserving magnitudes for MSE
        pm = torch.clamp(predicted_move, -10.0, 10.0)
        tm = torch.clamp(target_move, -10.0, 10.0)
        ps = torch.clamp(predicted_state, -10.0, 10.0)
        ts = torch.clamp(target_state, -10.0, 10.0)

        # MSE losses
        move_loss = F.mse_loss(pm, tm)
        state_loss = F.mse_loss(ps, ts)

        # Tiny L2 on predictions to keep them bounded
        move_reg = (pm.pow(2).mean()) * 1e-6
        state_reg = (ps.pow(2).mean()) * 1e-6

        total_loss = (self.move_weight * (move_loss + move_reg) +
                      self.state_weight * (state_loss + state_reg))
        total_loss = torch.clamp(total_loss, 0.0, 100.0)

        return total_loss, {
            'move_loss': move_loss.detach(),
            'state_loss': state_loss.detach(),
            'move_reg': move_reg.detach(),
            'state_reg': state_reg.detach(),
            'total_loss': total_loss.detach(),
        }


class SequentialHiveLLMPlayer(nn.Module):
    """Sequential HIVE LLM player with historical context"""
    
    def __init__(self, config: SequentialHiveLLMConfig):
        super().__init__()
        self.config = config

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        # Ensure a real PAD token distinct from EOS
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # Choose safe compute dtype: prefer BF16, else FP32 (avoid FP16 compute)
        self.compute_dtype = bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        # Initialize quantization config
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self.compute_dtype,
        )

        # Make attention numerics boring (stable)
        self.base_model.config.use_cache = False
        if hasattr(self.base_model, "generation_config"):
            self.base_model.generation_config.use_cache = False
        try:
            self.base_model.config.attn_implementation = "eager"
        except Exception:
            pass
        # Remove known odd/unknown keys if present to reduce warnings
        try:
            if hasattr(self.base_model.config, "attn_factor"):
                delattr(self.base_model.config, "attn_factor")
        except Exception:
            pass

        # Resize embeddings after tokenizer changes
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        print("PAD:", self.tokenizer.pad_token_id, "EOS:", self.tokenizer.eos_token_id)
        # Add special tokens
        self._add_special_tokens()

        # Prepare model for k-bit training
        if config.use_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        # Re-assert no caching after potential internal changes
        try:
            self.base_model.config.use_cache = False
            if hasattr(self.base_model, "generation_config") and self.base_model.generation_config is not None:
                self.base_model.generation_config.use_cache = False
        except Exception:
            pass

        # Initialize LoRA
        self._setup_lora()
        # Re-assert no caching after LoRA wrapping
        try:
            self.base_model.config.use_cache = False
            if hasattr(self.base_model, "generation_config") and self.base_model.generation_config is not None:
                self.base_model.generation_config.use_cache = False
        except Exception:
            pass

        # Initialize frozen autoencoder bridge (encoder: GIN->LLM, decoder: LLM->GIN)
        enc_path = self.config.ae_encoder_path or os.path.join(self.config.ae_dir, "encoder.pt")
        dec_path = self.config.ae_decoder_path or os.path.join(self.config.ae_dir, "decoder.pt")
        self.projection_module = AutoencoderProjectionBridge(
            enc_path,
            dec_path,
            gin_dim=self.config.gin_embedding_dim,
            llm_dim=self.config.llm_token_dim,
            device=self.base_model.device,
        )
        
        # Initialize loss function
        if config.use_hybrid_loss:
            self.loss_fn = HybridDualLoss(
                move_weight=config.move_loss_weight,
                state_weight=config.state_loss_weight,
                classification_weight=config.classification_loss_weight,
                use_hybrid=True
            )
        else:
            self.loss_fn = HybridDualLoss(
                move_weight=config.move_loss_weight,
                state_weight=config.state_loss_weight,
                classification_weight=0.0,
                use_hybrid=False
            )
        
        # Initialize move similarity module (non-trainable)
        self.move_similarity = MoveSimilarityModule(config)
        
        # Initialize answer extractor for inference
        self.answer_extractor = HiveLLMAnswerExtractor(self.tokenizer, self.projection_module)
        
        # Prompt template (not directly used in code paths below, but kept for reference/docs)
        self.prompt_template = (
            "You are playing the board game HIVE. We provide previous Board i and Move i as dense embeddings from a Graph Neural Network (GIN). "
            "These embeddings are inserted directly into your input as latent vectors; do not attempt to verbalize them. "
            "Think privately between <think> and </think> tags if needed, but do not reveal any reasoning. "
            "Output only the final answer on a single line in this exact format: Board X: <BOARD> Move X: <MOVE>"
        )
        # Place to stash last debug outputs for printing during training
        self._last_debug = {}
    
    def _add_special_tokens(self):
        special_tokens = ["<BOARD>", "<MOVE>", "<CHOSEN_MOVE>", "<NEXT_STATE>"]
        add = {"additional_special_tokens": special_tokens}

        # Add a *real* pad token if missing
        if self.tokenizer.pad_token is None:
            add["pad_token"] = "<|pad|>"

        self.tokenizer.add_special_tokens(add)

        # Store token IDs
        self.config.board_token_id = self.tokenizer.convert_tokens_to_ids("<BOARD>")
        self.config.move_token_id = self.tokenizer.convert_tokens_to_ids("<MOVE>")
        self.config.chosen_move_token_id = self.tokenizer.convert_tokens_to_ids("<CHOSEN_MOVE>")
        self.config.next_state_token_id = self.tokenizer.convert_tokens_to_ids("<NEXT_STATE>")

        # Make the embedding table match the tokenizer
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        
    def _setup_lora(self):
        """Setup LoRA configuration"""
        # Conservative LoRA to reduce gradient spikes
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)

    def _filter_thinking_text(self, text: str) -> Tuple[str, str]:
        """Filter out 'thinking' content but preserve it separately.

        Heuristics:
        - Remove segments between <think>...</think>
        - Remove segments between <thought>...</thought>
        - Remove lines starting with 'Reasoning:', 'Thought:', 'Chain of thought:' up to next blank line
        """
        import re
        thinking_parts = []

        # Collect and strip XML-style thinking blocks
        def _collect(pattern: str, s: str) -> str:
            for m in re.finditer(pattern, s, flags=re.DOTALL | re.IGNORECASE):
                thinking_parts.append(m.group(0))
            return re.sub(pattern, "", s, flags=re.DOTALL | re.IGNORECASE)

        filtered = text
        filtered = _collect(r"<\s*think\s*>.*?<\s*/\s*think\s*>", filtered)
        filtered = _collect(r"<\s*thought\s*>.*?<\s*/\s*thought\s*>", filtered)

        # Remove common prefixes denoting reasoning paragraphs
        lines = []
        buf_think = []
        in_reason = False
        for line in filtered.splitlines():
            if re.match(r"\s*(Reasoning:|Thought:|Chain of thought:)", line, flags=re.IGNORECASE):
                in_reason = True
                buf_think.append(line)
                continue
            if in_reason:
                if line.strip() == "":
                    # end of block on blank line
                    thinking_parts.append("\n".join(buf_think))
                    buf_think = []
                    in_reason = False
                else:
                    buf_think.append(line)
                continue
            lines.append(line)
        if buf_think:
            thinking_parts.append("\n".join(buf_think))

        clean = "\n".join(lines)
        thinking = "\n\n".join([p.strip() for p in thinking_parts if p.strip()])
        return clean.strip(), thinking

    def _create_prompt_for_generation(
        self,
        context_board_embeddings: torch.Tensor,
        context_move_embeddings: torch.Tensor,
        current_board_embedding: torch.Tensor,
        legal_move_embeddings: torch.Tensor,
        legal_move_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create the prompt embeddings like the soft prompt, but WITHOUT the output placeholders.

        Returns:
            input_embeds: [B, T, D]
            attention_mask: [B, T]
            next_idx: [B] tensor with the next index number used in labels
        """
        batch_size, seq_len = context_board_embeddings.shape[:2]

        # Project history into LLM space (same as before)
        projected_context_boards = []
        projected_context_moves = []
        for i in range(seq_len):
            board_emb = context_board_embeddings[:, i, :, :]
            move_emb = context_move_embeddings[:, i, :, :]
            projected_context_boards.append(self.projection_module.state_projection(board_emb))
            projected_context_moves.append(self.projection_module.move_projection.forward_single(move_emb))

        embed_layer = self.base_model.get_input_embeddings()
        device = current_board_embedding.device
        # Ensure projected context vectors match embedding table dtype/device
        proj_device = embed_layer.weight.device
        proj_dtype = embed_layer.weight.dtype
        for i in range(len(projected_context_boards)):
            projected_context_boards[i] = projected_context_boards[i].to(device=proj_device, dtype=proj_dtype)
            projected_context_moves[i] = projected_context_moves[i].to(device=proj_device, dtype=proj_dtype)


        batch_embeds: List[torch.Tensor] = []
        batch_masks: List[torch.Tensor] = []
        next_indices: List[int] = []

        for b in range(batch_size):
            segs: List[torch.Tensor] = []

            # Optional BOS
            if getattr(self.tokenizer, 'bos_token_id', None) is not None:
                # Use embedding table directly to avoid extra ops and ensure dtype/device
                bos = embed_layer.weight[self.tokenizer.bos_token_id].unsqueeze(0).to(device=proj_device, dtype=proj_dtype)
                segs.append(bos)

            # Intro and context
            intro = (
                "You are playing the board game HIVE. We provide previous Board i and Move i as dense embeddings from a Graph Neural Network (GIN). "
                "Each 'Board i: X Move i: X', X is a latent vector, it represents board embeddings and move embeddings; do not try to decode it into text.\n"
                "These are the previous board states and moves made by players who alternate:\n"
            )
            t = self.tokenizer(intro, return_tensors="pt", add_special_tokens=False)
            e = embed_layer(t.input_ids.to(device))[0]
            segs.append(e)

            # History
            for step in range(seq_len):
                lbl = f"Board {step}: "
                t = self.tokenizer(lbl, return_tensors="pt", add_special_tokens=False)
                e = embed_layer(t.input_ids.to(device))[0]
                segs.append(e)
                segs.append(projected_context_boards[step][b:b+1])

                lbl = f" Move {step}: "
                t = self.tokenizer(lbl, return_tensors="pt", add_special_tokens=False)
                e = embed_layer(t.input_ids.to(device))[0]
                segs.append(e)
                segs.append(projected_context_moves[step][b:b+1])

                if step < seq_len - 1:
                    t = self.tokenizer("\n", return_tensors="pt", add_special_tokens=False)
                    e = embed_layer(t.input_ids.to(device))[0]
                    segs.append(e)

            # Instruction and output format guidance (silent reasoning)
            guidance = (
                "\n\nPredict the next board state and move. You may think privately between <think> and </think> tags,"
                " but DO NOT reveal any reasoning in your visible output.\n"
                "Output ONLY one line in this exact format: Board: <BOARD> Move: <MOVE>\n"
                "Where <BOARD> and <MOVE> are placeholders for the latent embeddings to be inserted directly.\n"
                "You need to understand which move can be beneficial for the player and how it changes the game state.\n"                
                "Example of hidden reasoning: <think> I consider prior placements and likely next move... </think>\n"
                "Example final answer: Board: <BOARD> Move: <MOVE>\n"
            )
            t = self.tokenizer(guidance, return_tensors="pt", add_special_tokens=False)
            e = embed_layer(t.input_ids.to(device))[0]
            segs.append(e)

            # Start of answer line WITHOUT placeholders or indices
            prefix = "Board: "
            t = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
            e = embed_layer(t.input_ids.to(device))[0]
            segs.append(e)

            # Continue with Move label so model completes with tokens (include index to enforce format)
            t = self.tokenizer("Move: ", return_tensors="pt", add_special_tokens=False)
            e = embed_layer(t.input_ids.to(device))[0]
            # We'll not add "{next_idx}: " here to avoid forcing exact continuation; allow the model to write it
            # but we do add a space for natural continuation
            segs.append(e)

            seq = torch.cat(segs, dim=0)
            mask = torch.ones(seq.shape[0], dtype=torch.long, device=device)
            batch_embeds.append(seq)
            batch_masks.append(mask)
            # Maintain API compatibility: we no longer use indices, store 0
            next_indices.append(0)

        # Left-pad to max length
        max_len = max(s.shape[0] for s in batch_embeds)
        pad_tok = None
        if self.tokenizer.pad_token_id is not None:
            pad_tok = embed_layer.weight[self.tokenizer.pad_token_id].unsqueeze(0).to(device=proj_device, dtype=proj_dtype)

        padded: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for (seq, m) in zip(batch_embeds, batch_masks):
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                if pad_tok is not None:
                    pad_emb = pad_tok.repeat(pad_len, 1)
                else:
                    pad_emb = torch.zeros(pad_len, seq.shape[1], dtype=seq.dtype, device=device)
                seq = torch.cat([pad_emb, seq], dim=0)
                m = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device), m], dim=0)
            padded.append(seq)
            masks.append(m)

        input_embeds = torch.stack(padded, dim=0)
        attention_mask = torch.stack(masks, dim=0)
        next_idx_tensor = torch.tensor(next_indices, dtype=torch.long, device=device)
        # Debug: final input summary

        return input_embeds, attention_mask, next_idx_tensor

    def _generate_and_extract_vectors(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list, list, list]:
        """Run generate, parse for <BOARD>/<MOVE>, and return their last-hidden vectors.

        Returns:
            predicted_state_llm: [B, D]
            predicted_move_llm: [B, D]
            raw_texts: list[str] per batch
            clean_texts: list[str] per batch
            found_flags: list[Tuple[bool, bool]] indicating if (board_found, move_found)
        """
        device = input_embeds.device
        # Temporarily disable gradient checkpointing around generate to avoid cache warnings
        was_gc = getattr(self.base_model, "is_gradient_checkpointing", False)
        can_disable_gc = hasattr(self.base_model, "gradient_checkpointing_disable")
        can_enable_gc = hasattr(self.base_model, "gradient_checkpointing_enable")
        if was_gc and can_disable_gc:
            try:
                self.base_model.gradient_checkpointing_disable()
            except Exception:
                pass
        try:
            gen_kwargs = dict(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            if self.config.do_sample:
                gen_kwargs.update({
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                })
            gen_out = self.base_model.generate(**gen_kwargs)
        finally:
            if was_gc and can_enable_gc:
                try:
                    self.base_model.gradient_checkpointing_enable()
                except Exception:
                    pass

        gen_ids = gen_out.sequences  # [B, Lg]
        embed_layer = self.base_model.get_input_embeddings()
        gen_embeds = embed_layer(gen_ids.to(device))  # [B, Lg, D]

        # Re-run a forward with concatenated context + generated ids to obtain hidden states
        full_embeds = torch.cat([input_embeds, gen_embeds], dim=1)
        full_mask = torch.cat([
            attention_mask,
            torch.ones(gen_ids.size(0), gen_ids.size(1), dtype=attention_mask.dtype, device=device)
        ], dim=1)

        full_out = self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = full_out.hidden_states[-1]  # [B, Tctx+Lg, D]

        B = gen_ids.size(0)
        Tctx = input_embeds.size(1)
        D = last_hidden.size(-1)

        board_vecs = torch.zeros(B, D, device=device)
        move_vecs = torch.zeros(B, D, device=device)
        raw_texts = []
        clean_texts = []
        found_flags = []

        for b in range(B):
            ids = gen_ids[b].tolist()
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            raw_texts.append(text)
            clean, thinking = self._filter_thinking_text(text)
            # Only keep lines that look like final answers; hide everything else from console
            visible = "".join([ln.strip() for ln in clean.splitlines() if ("Board" in ln and "Move" in ln)])
            if not visible:
                # fallback to the clean text (which has no explicit <think> blocks) but still likely contains narrative; keep minimal
                visible = clean.strip()
            clean_texts.append(visible)

            # Find first occurrences of the special tokens in the generated ids
            board_id = self.config.board_token_id
            move_id = self.config.move_token_id
            try:
                bpos = ids.index(board_id) if board_id in ids else -1
            except ValueError:
                bpos = -1
            try:
                mpos = ids.index(move_id) if move_id in ids else -1
            except ValueError:
                mpos = -1

            board_found = bpos >= 0
            move_found = mpos >= 0
            found_flags.append((board_found, move_found))

            if board_found:
                board_vecs[b] = last_hidden[b, Tctx + bpos, :]
            else:
                # fallback: last token vector
                board_vecs[b] = last_hidden[b, -1, :]

            if move_found:
                move_vecs[b] = last_hidden[b, Tctx + mpos, :]
            else:
                # fallback: last token vector
                move_vecs[b] = last_hidden[b, -1, :]

        # Stash for external prints
        self._last_debug = {
            'raw_texts': raw_texts,
            'clean_texts': clean_texts,
            'found_flags': found_flags,
        }

        return board_vecs, move_vecs, raw_texts, clean_texts, found_flags

    @torch.no_grad()
    def debug_generate_text(
        self,
        context_board_embeddings: torch.Tensor,
        context_move_embeddings: torch.Tensor,
        current_board_embedding: torch.Tensor,
        legal_move_embeddings: torch.Tensor,
        legal_move_masks: torch.Tensor,
    ) -> Tuple[str, str, Tuple[bool, bool]]:
        """Generate a short answer and return (raw_text, cleaned_text, found_flags[0])."""
        # Temporarily switch to eval to avoid any training-mode layers requiring batch > 1
        prev_training_self = self.training
        prev_training_base = self.base_model.training
        try:
            self.eval()
            self.base_model.eval()

            # If configured to use free generation, use the generate+extract path.
            if self.config.use_free_generation:
                input_embeds, attention_mask, _ = self._create_prompt_for_generation(
                    context_board_embeddings,
                    context_move_embeddings,
                    current_board_embedding,
                    legal_move_embeddings,
                    legal_move_masks,
                )
                _, _, raw_texts, clean_texts, flags = self._generate_and_extract_vectors(input_embeds, attention_mask)
                if raw_texts:
                    return raw_texts[0], clean_texts[0], flags[0] if flags else (False, False)
                return "", "", (False, False)

            # Otherwise, use the placeholder (soft-prompt) path and extract hidden states directly.
            input_embeds, attention_mask, board_pos, move_pos = self._create_sequential_soft_prompt(
                context_board_embeddings,
                context_move_embeddings,
                current_board_embedding,
                legal_move_embeddings,
                legal_move_masks,
            )
            outputs = self.base_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            bsz = last_hidden.size(0)
            rng = torch.arange(bsz, device=last_hidden.device)
            predicted_state = last_hidden[rng, board_pos, :]
            predicted_move = last_hidden[rng, move_pos, :]



            # Map predicted LLM vectors back to GIN space for human-checks and move-similarity
            try:
                # predicted_move/state: [B, Dllm]
                # Convert to GIN using decoder/projection inverse
                pred_move_gin = self.projection_module.move_projection_inverse(predicted_move)
                pred_state_gin = self.projection_module.state_projection_inverse(predicted_state)

                # Legal moves: [B, N, 1, Dgin] or [B, N, Dgin]
                # We'll use the first batch entry for printing
                legal_moves_batch = legal_move_embeddings[0]
                legal_mask_batch = legal_move_masks[0]
                # Ensure shapes match expected for move_similarity
                if legal_moves_batch.dim() == 2:
                    legal_moves_for_sim = legal_moves_batch
                else:
                    legal_moves_for_sim = legal_moves_batch.squeeze(1)

                best_idx, sim = self.move_similarity.find_closest_move(predicted_move[0], legal_moves_for_sim, legal_mask_batch, self.projection_module)
                print(f"[DEBUG] Closest legal move index (batch0) = {best_idx}, score={sim:.6f}")
            except Exception as _e:
                print(f"[DEBUG] mapping to gin or similarity failed: {_e}")

            raw_texts = ["<PLACEHOLDER_PATH>" for _ in range(bsz)]
            clean_texts = ["<PLACEHOLDER_PATH>" for _ in range(bsz)]
            flags = [(True, True) for _ in range(bsz)]
            return raw_texts[0], clean_texts[0], flags[0]
        finally:
            # Restore training state
            try:
                if prev_training_self:
                    self.train()
                else:
                    self.eval()
                if prev_training_base:
                    self.base_model.train()
                else:
                    self.base_model.eval()
            except Exception:
                pass
        
    def _create_sequential_soft_prompt(self, 
                                 context_board_embeddings: torch.Tensor,
                                 context_move_embeddings: torch.Tensor,
                                 current_board_embedding: torch.Tensor,
                                 legal_move_embeddings: torch.Tensor,
                                 legal_move_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sequential soft prompt with historical context (no legal moves shown)
        
        Returns:
            input_embeds: [batch_size, total_seq_len, llm_token_dim]
            attention_mask: [batch_size, total_seq_len]
            board_pos: [batch_size] index of the <BOARD> placeholder in sequence
            move_pos: [batch_size] index of the <MOVE> placeholder in sequence
        """
        batch_size, seq_len = context_board_embeddings.shape[:2]

        # Project embeddings to LLM space
        projected_context_boards = []
        projected_context_moves = []
        for i in range(seq_len):
            board_emb = context_board_embeddings[:, i, :, :]  # [B, 1, gin]
            move_emb = context_move_embeddings[:, i, :, :]    # [B, 1, mov]
            projected_context_boards.append(self.projection_module.state_projection(board_emb))      # [B, D]
            projected_context_moves.append(self.projection_module.move_projection.forward_single(move_emb))  # [B, D]

        # Ensure projections match embedding table dtype/device
        _embed_layer_for_proj = self.base_model.get_input_embeddings()
        proj_device = _embed_layer_for_proj.weight.device
        proj_dtype = _embed_layer_for_proj.weight.dtype
        for i in range(len(projected_context_boards)):
            projected_context_boards[i] = projected_context_boards[i].to(device=proj_device, dtype=proj_dtype)
            projected_context_moves[i] = projected_context_moves[i].to(device=proj_device, dtype=proj_dtype)

    # NOTE: The new prompt format does NOT include the explicit current board line.
    # We therefore do not append the projected current board into the prompt.

        # Token embeddings
        embed_layer = self.base_model.get_input_embeddings()
        device = current_board_embedding.device

        batch_embeds: List[torch.Tensor] = []
        batch_masks: List[torch.Tensor] = []
        board_positions: List[int] = []
        move_positions: List[int] = []

        for b in range(batch_size):
            segs: List[torch.Tensor] = []
            cur_len = 0

            # Optional BOS
            if getattr(self.tokenizer, 'bos_token_id', None) is not None:
                bos = _embed_layer_for_proj.weight[self.tokenizer.bos_token_id].unsqueeze(0).to(device=proj_device, dtype=proj_dtype)
                segs.append(bos)
                cur_len += bos.shape[0]

            # Intro text
            intro = (
                "You are playing the board game HIVE. We provide previous Board i and Move i as dense embeddings from a Graph Neural Network (GIN). "
                "Each 'Board i: X Move i: X', X is a latent vector, it represents board embeddings and move embeddings; do not try to decode it into text.\n"
                "These are the previous board states and moves made by players who alternate:\n"
            )
            t = self.tokenizer(intro, return_tensors="pt", add_special_tokens=False)
            e = embed_layer(t.input_ids.to(device))[0]
            segs.append(e); cur_len += e.shape[0]

            # History
            for step in range(seq_len):
                # Board label
                lbl = f"Board {step}: "
                t = self.tokenizer(lbl, return_tensors="pt", add_special_tokens=False)
                e = embed_layer(t.input_ids.to(device))[0]
                segs.append(e); cur_len += e.shape[0]

                # Board soft token
                segs.append(projected_context_boards[step][b:b+1]); cur_len += 1

                # Move label
                lbl = f" Move {step}: "
                t = self.tokenizer(lbl, return_tensors="pt", add_special_tokens=False)
                e = embed_layer(t.input_ids.to(device))[0]
                segs.append(e); cur_len += e.shape[0]

                # Move soft token
                segs.append(projected_context_moves[step][b:b+1]); cur_len += 1

                if step < seq_len - 1:
                    t = self.tokenizer("\n", return_tensors="pt", add_special_tokens=False)
                    e = embed_layer(t.input_ids.to(device))[0]
                    segs.append(e); cur_len += e.shape[0]

            # Instruction and placeholders (single line):
            # Instruction and output format guidance (silent reasoning)
            guidance = (
                "\n\nPredict the next board state and move. You may think privately between <think> and </think> tags,"
                " but DO NOT reveal any reasoning in your visible output.\n"
                "Output ONLY one line in this exact format: Board: <BOARD> Move: <MOVE>\n"
                "Where <BOARD> and <MOVE> are placeholders for the latent embeddings to be inserted directly.\n"
                "You need to understand which move can be beneficial for the player and how it changes the game state.\n"                
                "Example of hidden reasoning: <think> I consider prior placements and likely next move... </think>\n"
                "Example final answer: Board: <BOARD> Move: <MOVE>\n"
            )
            t = self.tokenizer(guidance, return_tensors="pt", add_special_tokens=False)
            e = embed_layer(t.input_ids.to(device))[0]
            segs.append(e); cur_len += e.shape[0]

            # "Board: "
            t = self.tokenizer("Board: ", return_tensors="pt", add_special_tokens=False)
            colon = embed_layer(t.input_ids.to(device))[0]
            segs.append(colon); cur_len += colon.shape[0]

            # Special <BOARD> token position to predict next state
            if self.config.board_token_id is not None:
                tok = _embed_layer_for_proj.weight[self.config.board_token_id].unsqueeze(0).to(device=proj_device, dtype=proj_dtype)
                segs.append(tok)
                board_positions.append(cur_len)
                cur_len += tok.shape[0]

            # " Move: "
            t = self.tokenizer(" Move: ", return_tensors="pt", add_special_tokens=False)
            e = embed_layer(t.input_ids.to(device))[0]
            segs.append(e); cur_len += e.shape[0]

            # Special <MOVE> token position
            if self.config.move_token_id is not None:
                tok = _embed_layer_for_proj.weight[self.config.move_token_id].unsqueeze(0).to(device=proj_device, dtype=proj_dtype)
                segs.append(tok)
                move_positions.append(cur_len)
                cur_len += tok.shape[0]

            # Concat
            seq = torch.cat(segs, dim=0)
            mask = torch.ones(seq.shape[0], dtype=torch.long, device=device)
            batch_embeds.append(seq)
            batch_masks.append(mask)

        # Left-pad to max length
        max_len = max(s.shape[0] for s in batch_embeds)
        padded: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        adj_board: List[int] = []
        adj_move: List[int] = []

        pad_tok = None
        if self.tokenizer.pad_token_id is not None:
            pad_tok = _embed_layer_for_proj.weight[self.tokenizer.pad_token_id].unsqueeze(0).to(device=proj_device, dtype=proj_dtype)

        for i, (seq, m) in enumerate(zip(batch_embeds, batch_masks)):
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                if pad_tok is not None:
                    pad_emb = pad_tok.repeat(pad_len, 1)
                else:
                    pad_emb = torch.zeros(pad_len, seq.shape[1], dtype=seq.dtype, device=device)
                seq = torch.cat([pad_emb, seq], dim=0)
                m = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device), m], dim=0)
                adj_board.append(board_positions[i] + pad_len)
                adj_move.append(move_positions[i] + pad_len)
            else:
                adj_board.append(board_positions[i])
                adj_move.append(move_positions[i])
            padded.append(seq)
            masks.append(m)

        input_embeds = torch.stack(padded, dim=0)
        attention_mask = torch.stack(masks, dim=0)
        board_pos = torch.tensor(adj_board, dtype=torch.long, device=device)
        move_pos = torch.tensor(adj_move, dtype=torch.long, device=device)

        return input_embeds, attention_mask, board_pos, move_pos

    def forward(self, 
                context_board_embeddings: torch.Tensor,
                context_move_embeddings: torch.Tensor,
                current_board_embedding: torch.Tensor,
                legal_move_embeddings: torch.Tensor,
                legal_move_masks: torch.Tensor,
                chosen_move_embedding: torch.Tensor,
                next_board_embedding: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with sequential context.

        If use_free_generation is True, let the model generate the answer line and
        extract the <BOARD>/<MOVE> token vectors from generated tokens. Otherwise, use placeholders.
        """
        if self.config.use_free_generation:
            # Build generation prompt (no placeholders)
            input_embeds, attention_mask, _ = self._create_prompt_for_generation(
                context_board_embeddings,
                context_move_embeddings,
                current_board_embedding,
                legal_move_embeddings,
                legal_move_masks,
            )

            if not torch.isfinite(input_embeds).all():
                raise ValueError("Non-finite values in input_embeds")

            # Generate and extract vectors corresponding to generated <BOARD>/<MOVE> tokens
            predicted_state, predicted_move, _, _, _ = self._generate_and_extract_vectors(
                input_embeds, attention_mask
            )
        else:
            # Original placeholder-based method
            input_embeds, attention_mask, board_pos, move_pos = self._create_sequential_soft_prompt(
                context_board_embeddings,
                context_move_embeddings,
                current_board_embedding,
                legal_move_embeddings,
                legal_move_masks,
            )

            if not torch.isfinite(input_embeds).all():
                raise ValueError("Non-finite values in input_embeds")

            outputs = self.base_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            if not torch.isfinite(last_hidden).all():
                raise ValueError("Non-finite values in hidden states")

            bsz = last_hidden.size(0)
            rng = torch.arange(bsz, device=last_hidden.device)
            predicted_state = last_hidden[rng, board_pos, :]
            predicted_move = last_hidden[rng, move_pos, :]

        # Compute targets in LLM space using the frozen encoder (GIN -> LLM)
        # chosen_move_embedding: [B, 1, Dgin] or [B, Dgin]
        if chosen_move_embedding.dim() == 3:
            tgt_move_llm = self.projection_module.move_projection.forward_single(chosen_move_embedding)
        else:
            tgt_move_llm = self.projection_module.move_projection.forward_single(chosen_move_embedding.unsqueeze(1))
        # next_board_embedding: [B, 1, Dgin] or [B, Dgin]
        tgt_state_llm = self.projection_module.state_projection(next_board_embedding)

        # Predicted vectors are already in LLM space; compute cosine loss directly there
        loss, loss_dict = self.loss_fn(predicted_move, tgt_move_llm, predicted_state, tgt_state_llm)
        return loss, loss_dict
    
    @torch.no_grad()
    def generate_move(
        self,
        context_board_embeddings: torch.Tensor,
        context_move_embeddings: torch.Tensor,
        current_board_embedding: torch.Tensor,
        legal_move_embeddings: torch.Tensor,
        legal_move_masks: torch.Tensor,
    ):
        self.eval()
        if self.config.use_free_generation:
            # Use generation path and extract generated tokens
            input_embeds, attention_mask, _ = self._create_prompt_for_generation(
                context_board_embeddings,
                context_move_embeddings,
                current_board_embedding,
                legal_move_embeddings,
                legal_move_masks,
            )
            pred_state_batch, pred_move_batch, raw_texts, clean_texts, flags = self._generate_and_extract_vectors(
                input_embeds, attention_mask
            )
            pred_state = pred_state_batch[0]
            pred_move = pred_move_batch[0]
            # Optionally stash last outputs for caller
            self._last_debug = {
                'raw_texts': raw_texts,
                'clean_texts': clean_texts,
                'found_flags': flags,
            }
        else:
            input_embeds, attention_mask, board_pos, move_pos = self._create_sequential_soft_prompt(
                context_board_embeddings,
                context_move_embeddings,
                current_board_embedding,
                legal_move_embeddings,
                legal_move_masks,
            )
            outputs = self.base_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pred_state = last_hidden[0, board_pos.item(), :]
            pred_move = last_hidden[0, move_pos.item(), :]

        legal_moves_s = legal_move_embeddings.squeeze(0)
        legal_masks_s = legal_move_masks.squeeze(0)
        best_idx, sim = self.move_similarity.find_closest_move(
            pred_move, legal_moves_s, legal_masks_s, self.projection_module
        )
        return best_idx, float(sim), pred_move, pred_state



class SequentialHiveLLMTrainer:
    """Training pipeline for sequential HIVE LLM"""

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for sequential samples"""
        # Extract components
        context_board_embeddings = [item['context_board_embeddings'] for item in batch]
        context_move_embeddings = [item['context_move_embeddings'] for item in batch]
        current_board_embeddings = [item['current_board_embedding'] for item in batch]
        legal_move_embeddings_list = [item['legal_move_embeddings'] for item in batch]
        legal_move_texts_list = [item['legal_move_texts'] for item in batch]
        chosen_move_indices = [item['chosen_move_idx'] for item in batch]
        chosen_move_embeddings = [item['chosen_move_embedding'] for item in batch]
        next_board_embeddings = [item['next_board_embedding'] for item in batch]
        sequence_lengths = [item['sequence_length'] for item in batch]

        # Stack fixed-size tensors
        context_board_embeddings = torch.stack(context_board_embeddings, dim=0)
        context_move_embeddings = torch.stack(context_move_embeddings, dim=0)
        current_board_embeddings = torch.stack(current_board_embeddings, dim=0)
        chosen_move_embeddings = torch.stack(chosen_move_embeddings, dim=0)
        next_board_embeddings = torch.stack(next_board_embeddings, dim=0)
        chosen_move_indices = torch.tensor(chosen_move_indices, dtype=torch.long)

        # Handle variable-length legal moves
        max_legal_moves = max(moves.size(0) for moves in legal_move_embeddings_list)
        padded_legal_move_embeddings = []
        padded_legal_move_texts = []
        legal_move_masks = []

        for moves, texts in zip(legal_move_embeddings_list, legal_move_texts_list):
            num_moves, _, move_dim = moves.shape
            padded_moves = torch.zeros(max_legal_moves, 1, move_dim)
            padded_moves[:num_moves] = moves
            padded_legal_move_embeddings.append(padded_moves)

            padded_texts = texts + ["<PAD>"] * (max_legal_moves - len(texts))
            padded_legal_move_texts.append(padded_texts)

            mask = torch.zeros(max_legal_moves, dtype=torch.bool)
            mask[:num_moves] = True
            legal_move_masks.append(mask)

        legal_move_embeddings = torch.stack(padded_legal_move_embeddings, dim=0)
        legal_move_masks = torch.stack(legal_move_masks, dim=0)

        return {
            'context_board_embeddings': context_board_embeddings,
            'context_move_embeddings': context_move_embeddings,
            'current_board_embedding': current_board_embeddings,
            'legal_move_embeddings': legal_move_embeddings,
            'legal_move_texts': padded_legal_move_texts,
            'legal_move_masks': legal_move_masks,
            'chosen_move_idx': chosen_move_indices,
            'chosen_move_embedding': chosen_move_embeddings,
            'next_board_embedding': next_board_embeddings,
            'sequence_lengths': torch.tensor(sequence_lengths, dtype=torch.long)
        }

    def __init__(self, config: SequentialHiveLLMConfig, data_dir: str):
        self.config = config
        self.data_dir = data_dir

        # Initialize model
        self.model = SequentialHiveLLMPlayer(config)

        # Setup device and move model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # The base model is already on GPU due to device_map="auto"
        base_model_device = next(self.model.base_model.parameters()).device
        self.model.projection_module.to(base_model_device)

        print(f"Training device: {self.device}")
        print(f"Base model device: {base_model_device}")
        print(f"Projection module device: {next(self.model.projection_module.parameters()).device}")

        # Initialize datasets
        self.train_dataset = SequentialHiveDataset(data_dir, "train", config)
        self.val_dataset = SequentialHiveDataset(data_dir, "validation", config)

        # Initialize data loaders with custom collate function
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

        # Initialize optimizers
        self.setup_optimizers()

        # Mixed precision hygiene: use GradScaler only for fp16; prefer bf16 autocast otherwise
        model_compute = getattr(self.model, 'compute_dtype', torch.float32)
        self.scaler = GradScaler() if (config.use_mixed_precision and model_compute == torch.float16) else None

        # Initialize tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.consecutive_nan_steps = 0

        # Metrics logging setup (lightweight CSV; no extra forward passes)
        os.makedirs("logs", exist_ok=True)
        self.train_metrics_path = os.path.join("logs", "train_metrics.csv")
        self.val_metrics_path = os.path.join("logs", "val_metrics.csv")
        self.epoch_metrics_path = os.path.join("logs", "epoch_metrics.csv")
        self._train_csv_initialized = os.path.exists(self.train_metrics_path)
        self._val_csv_initialized = os.path.exists(self.val_metrics_path)
        self._epoch_csv_initialized = os.path.exists(self.epoch_metrics_path)
        if not self._train_csv_initialized:
            try:
                with open(self.train_metrics_path, "w", encoding="utf-8") as f:
                    f.write(
                        "step,epoch,loss,move_loss,state_loss,lr_lora,lr_base,grad_norm,sec_per_step,gpu_mem_mb\n"
                    )
                self._train_csv_initialized = True
            except Exception:
                pass
        if not self._val_csv_initialized:
            try:
                with open(self.val_metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,val_loss\n")
                self._val_csv_initialized = True
            except Exception:
                pass
        if not self._epoch_csv_initialized:
            try:
                with open(self.epoch_metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,split,total_loss,move_mse,state_mse\n")
                self._epoch_csv_initialized = True
            except Exception:
                pass

    def setup_optimizers(self):
        """Setup optimizers with different learning rates for different components"""
        lora_params: List[torch.nn.Parameter] = []
        base_other: List[torch.nn.Parameter] = []
        for n, p in self.model.base_model.named_parameters():
            if not p.requires_grad:
                continue
            if 'lora_' in n:
                lora_params.append(p)
            else:
                base_other.append(p)

        # Autoencoder bridge is frozen by request; no head params to optimize

        self.optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": 1e-4, "weight_decay": 0.0},
                # No projection head params (frozen AE)
                {"params": base_other, "lr": 5e-6, "weight_decay": 0.01},
            ],
            betas=(0.9, 0.999), eps=1e-8,
        )

        total_steps = max(1, len(self.train_loader) * self.config.num_epochs // max(1, self.config.gradient_accumulation_steps))
        warmup_steps = max(1, int(0.05 * total_steps))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    def train_epoch(self):
        """Train for one epoch with sequential context"""
        self.model.train()
        epoch_losses: List[float] = []
        epoch_move_mse: List[float] = []
        epoch_state_mse: List[float] = []

        progress_bar = tqdm(self.train_loader, desc="Training Sequential", leave=False, ncols=120, dynamic_ncols=False)
        last_step_time = time.perf_counter()
        last_epoch_idx = getattr(self, "_current_epoch", 0)

        for step, batch in enumerate(progress_bar):
            if self.consecutive_nan_steps >= self.config.nan_tolerance:
                logger.error(f"Too many consecutive NaN steps ({self.consecutive_nan_steps}), stopping training")
                break

            device = next(self.model.base_model.parameters()).device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            input_has_nan = False
            for key, tensor in batch.items():
                if torch.is_tensor(tensor) and torch.isnan(tensor).any():
                    logger.warning(f"NaN detected in input {key}, skipping batch")
                    input_has_nan = True
                    break
            if input_has_nan:
                continue

            if self.global_step == 0 and step == 0:
                try:
                    raw0, clean0, (bf0, mf0) = self.model.debug_generate_text(
                        batch['context_board_embeddings'][:1],
                        batch['context_move_embeddings'][:1],
                        batch['current_board_embedding'][:1],
                        batch['legal_move_embeddings'][:1],
                        batch['legal_move_masks'][:1],
                    )
                    msg0 = f"[warmup] LLM output (clean) -> [board={bf0}, move={mf0}]: " + ((clean0.replace("\n", " ") if clean0 else "<empty>"))
                    try:
                        progress_bar.write(msg0)
                    except Exception:
                        print(msg0, flush=True)
                    logger.info(msg0)
                    if raw0:
                        msg0r = "[warmup] LLM output (raw): " + raw0[:800].replace("\n", " ")
                        try:
                            progress_bar.write(msg0r)
                        except Exception:
                            print(msg0r, flush=True)
                        logger.info(msg0r)
                    try:
                        os.makedirs("logs", exist_ok=True)
                        with open(os.path.join("logs", "llm_debug.txt"), "a", encoding="utf-8") as f:
                            f.write("== WARMUP ==\n")
                            f.write(f"clean: {clean0}\n")
                            f.write(f"raw:   {raw0}\n\n")
                    except Exception:
                        pass
                except Exception:
                    pass

            try:
                if self.config.use_mixed_precision and self.scaler is not None:
                    with autocast():
                        loss, loss_dict = self.model(
                            batch['context_board_embeddings'],
                            batch['context_move_embeddings'],
                            batch['current_board_embedding'],
                            batch['legal_move_embeddings'],
                            batch['legal_move_masks'],
                            batch['chosen_move_embedding'],
                            batch['next_board_embedding']
                        )
                elif self.config.use_mixed_precision and getattr(self.model, 'compute_dtype', torch.float32) == torch.bfloat16 and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss, loss_dict = self.model(
                            batch['context_board_embeddings'],
                            batch['context_move_embeddings'],
                            batch['current_board_embedding'],
                            batch['legal_move_embeddings'],
                            batch['legal_move_masks'],
                            batch['chosen_move_embedding'],
                            batch['next_board_embedding']
                        )
                else:
                    loss, loss_dict = self.model(
                        batch['context_board_embeddings'],
                        batch['context_move_embeddings'],
                        batch['current_board_embedding'],
                        batch['legal_move_embeddings'],
                        batch['legal_move_masks'],
                        batch['chosen_move_embedding'],
                        batch['next_board_embedding']
                    )

                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                    self.consecutive_nan_steps += 1
                    logger.warning(f"NaN/Inf/Large loss detected at step {step} (value: {loss.item()}), skipping... ({self.consecutive_nan_steps}/{self.config.nan_tolerance})")
                    continue
                else:
                    self.consecutive_nan_steps = 0

            except Exception as e:
                logger.error(f"Error in forward pass at step {step}: {e}")
                self.consecutive_nan_steps += 1
                continue

            loss = loss / self.config.gradient_accumulation_steps

            try:
                if self.config.use_mixed_precision and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except Exception as e:
                logger.error(f"Error in backward pass at step {step}: {e}")
                self.optimizer.zero_grad()
                self.consecutive_nan_steps += 1
                continue

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.warning(f"NaN gradient detected in {name}")
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    self.optimizer.zero_grad()
                    self.consecutive_nan_steps += 1
                    continue

                if self.config.use_mixed_precision and self.scaler is not None:
                    try:
                        self.scaler.unscale_(self.optimizer)
                        grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.base_model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except Exception as e:
                        logger.warning(f"Mixed precision step failed: {e}, falling back to regular step")
                        grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.base_model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                else:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.base_model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.config.logging_steps == 0:
                    logger.info(f"Step {self.global_step}: {loss_dict}")
                    try:
                        lr_lora = None
                        lr_base = None
                        if isinstance(self.optimizer.param_groups, list) and len(self.optimizer.param_groups) >= 2:
                            lr_lora = float(self.optimizer.param_groups[0].get("lr", 0.0))
                            lr_base = float(self.optimizer.param_groups[1].get("lr", 0.0))
                        else:
                            lr_base = float(self.optimizer.param_groups[0].get("lr", 0.0))
                        now = time.perf_counter()
                        sec_per_step = now - last_step_time
                        last_step_time = now
                        gpu_mem_mb = float(torch.cuda.memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
                        with open(self.train_metrics_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"{self.global_step},{last_epoch_idx},{float(loss_dict['total_loss']):.6f},{float(loss_dict['move_loss']):.6f},{float(loss_dict['state_loss']):.6f},{'' if lr_lora is None else f'{lr_lora:.8f}'},{'' if lr_base is None else f'{lr_base:.8f}'},{float(grad_total_norm) if 'grad_total_norm' in locals() else ''},{sec_per_step:.6f},{gpu_mem_mb:.1f}\n"
                            )
                    except Exception:
                        pass

                if self.global_step % max(1, self.config.print_every_n_steps) == 0:
                    try:
                        k = 2 if batch['context_board_embeddings'].size(0) >= 2 else 1
                        raw, clean, (bf, mf) = self.model.debug_generate_text(
                            batch['context_board_embeddings'][:k],
                            batch['context_move_embeddings'][:k],
                            batch['current_board_embedding'][:k],
                            batch['legal_move_embeddings'][:k],
                            batch['legal_move_masks'][:k],
                        )
                        snippet_clean = (clean.replace("\n", " ") if clean else "<empty>")
                        snippet_raw = (raw[:800].replace("\n", " ") if raw else "<none>")
                        msg = (
                            f"LLM output @step {self.global_step} [board={bf}, move={mf}]\n"
                            f"  clean: {snippet_clean}\n"
                            f"  raw:   {snippet_raw}"
                        )
                        try:
                            progress_bar.write(msg)
                        except Exception:
                            print(msg, flush=True)
                        logger.info(msg)
                        try:
                            os.makedirs("logs", exist_ok=True)
                            with open(os.path.join("logs", "llm_debug.txt"), "a", encoding="utf-8") as f:
                                f.write(f"== STEP {self.global_step} ==\n")
                                f.write(f"clean: {clean}\n")
                                f.write(f"raw:   {raw}\n\n")
                        except Exception:
                            pass
                    except Exception as _e:
                        try:
                            progress_bar.write(f"LLM debug generation failed @step {self.global_step}: {_e}")
                        except Exception:
                            logger.info(f"LLM debug generation failed @step {self.global_step}: {_e}")

            # Accumulate per-batch metrics (scale back after GA)
            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            try:
                epoch_move_mse.append(float(loss_dict['move_loss']))
                epoch_state_mse.append(float(loss_dict['state_loss']))
            except Exception:
                pass

            safe_loss = loss.item() if not torch.isnan(loss) else 0.0
            safe_move_loss = loss_dict['move_loss'].item() if not torch.isnan(loss_dict['move_loss']) else 0.0
            safe_state_loss = loss_dict['state_loss'].item() if not torch.isnan(loss_dict['state_loss']) else 0.0
            progress_bar.set_postfix({
                'loss': f"{safe_loss:.4f}",
                'move_loss': f"{safe_move_loss:.4f}",
                'state_loss': f"{safe_state_loss:.4f}",
                'nan_count': self.consecutive_nan_steps
            })

        progress_bar.close()
        # Epoch-level logging summarizing MSE without extra compute
        train_total = np.mean(epoch_losses) if epoch_losses else float('inf')
        train_move = np.mean(epoch_move_mse) if epoch_move_mse else float('inf')
        train_state = np.mean(epoch_state_mse) if epoch_state_mse else float('inf')
        try:
            with open(self.epoch_metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{getattr(self, '_current_epoch', 0)},train,{train_total:.6f},{train_move:.6f},{train_state:.6f}\n")
        except Exception:
            pass
        logger.info(f"[Train Epoch {getattr(self, '_current_epoch', 0)}] total={train_total:.4f} move_mse={train_move:.4f} state_mse={train_state:.4f}")
        return train_total

    def validate(self):
        """Validation loop for sequential model"""
        self.model.eval()
        val_losses = []
        val_move_mse = []
        val_state_mse = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation Sequential", leave=False, ncols=120, dynamic_ncols=False)
            for batch in progress_bar:
                device = next(self.model.base_model.parameters()).device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                try:
                    if self.config.use_mixed_precision and self.scaler is not None:
                        with autocast():
                            loss, loss_dict = self.model(
                                batch['context_board_embeddings'],
                                batch['context_move_embeddings'],
                                batch['current_board_embedding'],
                                batch['legal_move_embeddings'],
                                batch['legal_move_masks'],
                                batch['chosen_move_embedding'],
                                batch['next_board_embedding']
                            )
                    elif self.config.use_mixed_precision and getattr(self.model, 'compute_dtype', torch.float32) == torch.bfloat16 and torch.cuda.is_available():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            loss, loss_dict = self.model(
                                batch['context_board_embeddings'],
                                batch['context_move_embeddings'],
                                batch['current_board_embedding'],
                                batch['legal_move_embeddings'],
                                batch['legal_move_masks'],
                                batch['chosen_move_embedding'],
                                batch['next_board_embedding']
                            )
                    else:
                        loss, loss_dict = self.model(
                            batch['context_board_embeddings'],
                            batch['context_move_embeddings'],
                            batch['current_board_embedding'],
                            batch['legal_move_embeddings'],
                            batch['legal_move_masks'],
                            batch['chosen_move_embedding'],
                            batch['next_board_embedding']
                        )
                    if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() <= 100.0:
                        val_losses.append(loss.item())
                        try:
                            val_move_mse.append(float(loss_dict['move_loss']))
                            val_state_mse.append(float(loss_dict['state_loss']))
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Error in validation: {e}")
                    continue
            progress_bar.close()

        val_mean = np.mean(val_losses) if val_losses else float('inf')
        val_move = np.mean(val_move_mse) if val_move_mse else float('inf')
        val_state = np.mean(val_state_mse) if val_state_mse else float('inf')
        # Write epoch-level validation metric
        try:
            with open(self.val_metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{getattr(self, '_current_epoch', 0)},{val_mean:.6f}\n")
        except Exception:
            pass
        # Also write to epoch_metrics.csv for a unified view
        try:
            with open(self.epoch_metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{getattr(self, '_current_epoch', 0)},val,{val_mean:.6f},{val_move:.6f},{val_state:.6f}\n")
        except Exception:
            pass
        logger.info(f"[Val   Epoch {getattr(self, '_current_epoch', 0)}] total={val_mean:.4f} move_mse={val_move:.4f} state_mse={val_state:.4f}")
        return val_mean

    def save_model(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        self.model.base_model.save_pretrained(os.path.join(path, "lora_adapter"))
    # Projection bridge (AE) is frozen and external; not saving here.
        with open(os.path.join(path, "config.json"), 'w') as f:
            config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_') and k not in ['lora_target_modules']}
            config_dict['lora_target_modules'] = self.config.lora_target_modules
            json.dump(config_dict, f, indent=2)
        self.model.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        if is_best:
            logger.info(f"Saved best sequential model to {path}")
        else:
            logger.info(f"Saved sequential checkpoint to {path}")

    def train(self):
        """Main training loop for sequential model"""
        logger.info("Starting sequential training...")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"Sequential context length: {self.config.sequence_length}")

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            # Track current epoch for logging
            self._current_epoch = epoch + 1
            self.consecutive_nan_steps = 0
            train_loss = self.train_epoch()
            if self.consecutive_nan_steps >= self.config.nan_tolerance:
                logger.error("Training stopped due to too many consecutive NaN losses")
                break
            logger.info(f"Training loss: {train_loss:.4f}")
            val_loss = self.validate()
            logger.info(f"Validation loss: {val_loss:.4f}")
            checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch + 1}")
            self.save_model(checkpoint_path)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.config.output_dir, "best_model")
                self.save_model(best_path, is_best=True)

        logger.info("Sequential training completed!")


def main():
    """Main training script for sequential HIVE LLM"""
    # Configuration
    config = SequentialHiveLLMConfig(
        base_model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        move_embedding_mode="difference",
        sequence_length=5,
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=8,
        use_free_generation=False,
        output_dir="models/LLM_1000_tournament_mse"
    )
    
    # Data directory (should contain sequential datasets)
    data_dir = "data/tournaments_1000"
    
    # Initialize trainer
    trainer = SequentialHiveLLMTrainer(config, data_dir)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()