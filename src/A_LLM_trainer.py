#!/usr/bin/env python3
"""
Simplified HIVE LLM architecture with model loading utilities.
Contains only the essential components for LLM and GNN model loading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import logging
import pickle
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterable
from dataclasses import dataclass, field
from torch import bfloat16
from torch.utils.data import Dataset, DataLoader
import numpy as np
from contextlib import nullcontext

# Attempt Unsloth import (gpt-oss optimized loader). Falls back gracefully if unavailable.
try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False

# Hugging Face and LoRA imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)

# Set up logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add tqdm for progress bars
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    logger.warning("tqdm not available - no progress bars will be shown")

# ---------------------------------------------------------------------------
# Windows note: Torch 2.8 + Unsloth may trigger torch.compile / inductor which
# requires a local C++ compiler (cl.exe) on Windows. If Build Tools are not
# installed this raises: RuntimeError: Compiler: cl is not found.
# We proactively disable torch.compile on Windows unless the user explicitly
# opts back in by setting TORCH_COMPILE_DISABLE=0 beforehand.
# ---------------------------------------------------------------------------
if os.name == 'nt' and os.environ.get("TORCH_COMPILE_DISABLE", "").lower() not in ("0", "false"):
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    logger.info("[init] Disabled torch.compile/inductor on Windows (set TORCH_COMPILE_DISABLE=0 to re-enable after installing MSVC Build Tools).")


@dataclass
class HiveLLMConfig:
    """Configuration for HIVE LLM"""
    
    # Model configuration
    # Switched default backbone to GPT-OSS via Unsloth. Override via CLI or config if needed.
    base_model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    gin_embedding_dim: int = 256
    llm_token_dim: int = 2880
    
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
    batch_size: int = 2
    use_cache: bool = False
    
    # Paths
    output_dir: str = "models/hive_llm"
    gnn_model_path: str = "src/models/pretrain_GIN_3.pt"
    # Cluster / centroid paths
    board_centroids_path: Optional[str] = "clustering_models/boards/cluster_centroids_kmeans_best.pkl"
    move_centroids_path: Optional[str] = "clustering_models/moves/cluster_centroids_kmeans_best.pkl"
    # Dataset caches (produced by sequential data generation step)
    train_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/train_sequential_cache.pkl"
    val_cache_path: Optional[str] = "data/sequential_hive_llm_dataset/validation_sequential_cache.pkl"
    # Cluster token settings
    board_cluster_token_prefix: str = "BCL"
    move_cluster_token_prefix: str = "MCL"
    add_eos_token: bool = True
    # Training hyperparameters
    epochs: int = 1
    grad_accumulation_steps: int = 1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 1
    eval_every: int = 200
    save_every: int = 0  # 0 disables periodic saves
    learning_rate: float = 5e-5  # (retain field above for backwards compat)
    lr_scheduler: str = "cosine"  # or linear
    seed: int = 42
    # ---- Cluster-token specific loss settings ----
    board_loss_weight: float = 1.0
    move_loss_weight: float = 1.0
    # If True: compute loss only on cluster tokens (board & move); otherwise standard causal LM across all tokens
    loss_on_cluster_tokens_only: bool = True
    # Optional cap for tokenized training sequence length (None = no extra truncation)
    max_train_sequence_length: Optional[int] = None
    # Unsloth / performance controls
    enable_gradient_checkpointing: bool = True  # enables Unsloth gradient checkpointing when available
    freeze_non_lora: bool = True  # ensure only LoRA adapter (and added projection) train
    suppress_unsloth_warnings: bool = False  # try to mute verbose Unsloth prints


def _load_centroids(path: Optional[str]) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # Accept dict forms
        if isinstance(obj, dict):
            for k in ['centroids', 'cluster_centroids', 'model_centroids']:
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).float()
        if torch.is_tensor(obj):
            return obj.float()
        if isinstance(obj, list):
            return torch.tensor(obj, dtype=torch.float32)
    except Exception as e:
        logger.warning(f"Failed to load centroids from {path}: {e}")
    return None


class GameClusterSequenceDataset(Dataset):
    """Wrap pickled sequential samples, turning cluster id sequences into text token sequences.

    Expected each sample dict to contain lists:
      'board_cluster_ids_sequence' : List[int] (or None)
      'chosen_move_cluster_ids_sequence' : List[int] (or None)
    We build an interleaved textual sequence: <BCL_i> <MCL_j> <BCL_i2> <MCL_j2> ...
    Optionally prepend BOS and append EOS (EOS only if config.add_eos_token True).
    """
    def __init__(
        self,
        samples: List[Dict],
        board_tokens: List[str],
        move_tokens: List[str],
        tokenizer,
        add_eos: bool = True,
        bos_token: Optional[str] = None,
    ):
        self.samples = samples
        self.board_tokens = board_tokens
        self.move_tokens = move_tokens
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.bos_token = bos_token or tokenizer.bos_token

        # Precompute mapping to ensure safe indexing
        self.num_board = len(board_tokens)
        self.num_move = len(move_tokens)

    def __len__(self):
        return len(self.samples)

    def build_text(self, sample: Dict) -> str:
        # Fetch sequences; avoid using `or []` because tensors / numpy arrays
        # with more than one element raise an error on boolean evaluation.
        b_seq = sample.get('board_cluster_ids_sequence', None)
        m_seq = sample.get('chosen_move_cluster_ids_sequence', None)

        # Normalize to Python lists for safe indexing & length ops.
        def _norm(x):
            if x is None:
                return []
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            # Some datasets might store as a single int (edge case)
            if isinstance(x, (int, np.integer)):
                return [int(x)]
            # Assume it's already an iterable of ints
            if isinstance(x, (list, tuple)):
                return list(x)
            try:
                return list(x)
            except Exception:
                return []

        b_seq = _norm(b_seq)
        m_seq = _norm(m_seq)
        # Interleave assuming lengths aligned (common). If lengths mismatch, clip to min.
        L = min(len(b_seq), len(m_seq))
        parts: List[str] = []
        for i in range(L):
            b = b_seq[i]
            m = m_seq[i]
            if 0 <= b < self.num_board:
                parts.append(self.board_tokens[b])
            if 0 <= m < self.num_move:
                parts.append(self.move_tokens[m])
        # Optionally include trailing final board if sequence longer by one
        if len(b_seq) == L + 1 and len(b_seq) > len(m_seq):
            b = b_seq[-1]
            if 0 <= b < self.num_board:
                parts.append(self.board_tokens[b])
        if self.add_eos and self.tokenizer.eos_token and (not parts or parts[-1] != self.tokenizer.eos_token):
            parts.append(self.tokenizer.eos_token)
        if self.bos_token:
            return f"{self.bos_token} " + " ".join(parts)
        return " ".join(parts)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = self.build_text(sample)
        enc = self.tokenizer(
            text,
            return_tensors='pt',
            padding=False,
            truncation=True,
        )
        input_ids = enc['input_ids'][0]
        attn_mask = enc['attention_mask'][0]
        # Standard causal LM: labels are shifted inside model; we just copy input_ids with -100 for padding (none expected)
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': labels,
            'raw_text': text,
        }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad to max length in batch
        max_len = max(x['input_ids'].size(0) for x in batch)
        
        # Get pad_token_id from first sample's tokenizer info or use 0 as fallback
        pad_token_id = 0  # Default fallback
        
        def pad(t: torch.Tensor, pad_value: int = 0):
            if t.size(0) == max_len:
                return t
            pad_len = max_len - t.size(0)
            return torch.cat([t, torch.full((pad_len,), fill_value=pad_value, dtype=t.dtype)], dim=0)
            
        input_ids = torch.stack([pad(b['input_ids'], pad_token_id) for b in batch])
        attention_mask = torch.stack([pad(b['attention_mask'], 0) for b in batch])
        labels = torch.stack([pad(b['labels'], -100) for b in batch])
        
        # Set label to -100 where attention_mask == 0 (padding positions)
        labels = labels.masked_fill(attention_mask == 0, -100)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class GNNLoader:
    """Utility class for loading GNN models"""
    
    @staticmethod
    def load_gnn_model(model_path: str, device: Optional[torch.device] = None) -> nn.Module:
        """Load a pretrained GNN model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GNN model not found at {model_path}")
        
        try:
            # Load the model state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model if it's wrapped in a dictionary
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                else:
                    model_state = checkpoint
            else:
                model_state = checkpoint
            
            logger.info(f"Successfully loaded GNN model from {model_path}")
            
            # Note: You'll need to instantiate the actual GNN model class here
            # This is a placeholder that returns the state dict
            return model_state
            
        except Exception as e:
            logger.error(f"Failed to load GNN model from {model_path}: {e}")
            raise


class HiveLLMModel(nn.Module):
    """Main HIVE LLM model with essential components.

    Now supports Unsloth GPT-OSS accelerated loading (preferred) while keeping
    a fallback to standard HuggingFace AutoModelForCausalLM when Unsloth is not
    installed or a non-unsloth model name is specified.
    """

    def __init__(self, config: HiveLLMConfig):
        super().__init__()
        self.config = config
        logger.info("Initializing HiveLLM model...")
        logger.info(f"Base model: {config.base_model_name}")
        logger.info(f"Using 4-bit quantization: {config.use_4bit}")
        
        self._load_backbone()

        # Projection from GIN embedding (centroid) -> LLM token embedding space
        logger.info("Setting up centroid projection layer...")
        self.centroid_projection = nn.Linear(config.gin_embedding_dim, self.base_model.get_input_embeddings().embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.centroid_projection.weight)

        # Placeholders for cluster tokens & centroids
        self.board_centroids: Optional[torch.Tensor] = None
        self.move_centroids: Optional[torch.Tensor] = None
        self.board_cluster_tokens: List[str] = []
        self.move_cluster_tokens: List[str] = []
        self.board_token_to_id: Dict[str, int] = {}
        self.move_token_to_id: Dict[str, int] = {}
        self.cluster_token_ids: List[int] = []  # all cluster token ids

        # Load (placeholder) GNN model state (not used directly here yet)
        logger.info("Loading GNN model...")
        self.gnn_loader = GNNLoader()
        try:
            self.gnn_model = self.gnn_loader.load_gnn_model(config.gnn_model_path)
        except Exception as e:
            logger.warning(f"Failed to load GNN model: {e}")
            self.gnn_model = None

        # Optionally auto-add cluster tokens if centroid paths exist
        logger.info("Adding cluster tokens...")
        self.try_add_cluster_tokens(
            config.board_centroids_path,
            config.move_centroids_path,
            initialize_embeddings=True,
        )
        logger.info("HiveLLM model initialized successfully")
        # Ensure LoRA adapter params match base model dtype (avoid BF16 vs FP32 addmm errors)
        self._align_lora_dtypes()
    
        
    def _load_backbone(self):
        name = self.config.base_model_name
        use_unsloth = name.startswith("unsloth/") and _HAS_UNSLOTH
        self.compute_dtype = (
            bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        )
        
        logger.info(f"Loading backbone model: {name}")
        logger.info(f"Using Unsloth: {use_unsloth}")
        logger.info(f"Compute dtype: {self.compute_dtype}")
        
        if use_unsloth:
            logger.info(f"Loading Unsloth FastLanguageModel backbone: {name}")
            # API churn handling: Prefer 'dtype' (new); fall back to 'torch_dtype' for older versions.
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=name,
                    max_seq_length=self.config.max_seq_length,
                    dtype=self.compute_dtype,
                    load_in_4bit=self.config.use_4bit,
                    full_finetuning=False,
                )
            except TypeError:
                logger.info("Unsloth from_pretrained rejected 'dtype'; retrying with 'torch_dtype'.")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=name,
                    max_seq_length=self.config.max_seq_length,
                    torch_dtype=self.compute_dtype,
                    load_in_4bit=self.config.use_4bit,
                    full_finetuning=False,
                )
            # Apply LoRA via Unsloth helper
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=self.config.lora_target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=("unsloth" if self.config.enable_gradient_checkpointing else False),
                random_state=self.config.seed,
                use_rslora=False,
                loftq_config=None,
            )
            self.base_model = model
            self.tokenizer = tokenizer
            if not self.config.enable_gradient_checkpointing and hasattr(self.base_model, 'gradient_checkpointing_disable'):
                try:
                    self.base_model.gradient_checkpointing_disable()
                except Exception:
                    pass
        else:
            if name.startswith("unsloth/") and not _HAS_UNSLOTH:
                logger.warning("Unsloth not installed; attempting standard HF load of unsloth model (may be slower). Install 'unsloth' for optimized path.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Setting up quantization configuration...")
            bnb_config = None
            if self.config.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=self.compute_dtype,
                    bnb_4bit_use_double_quant=self.config.use_nested_quant,
                )
            
            logger.info("Loading AutoModelForCausalLM...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=self.compute_dtype,
            )
            self.base_model.config.use_cache = self.config.use_cache
            if hasattr(self.base_model, "generation_config"):
                self.base_model.generation_config.use_cache = self.config.use_cache
                
            logger.info("Resizing token embeddings...")
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            
            if self.config.use_4bit:
                logger.info("Preparing model for k-bit training...")
                self.base_model = prepare_model_for_kbit_training(self.base_model)
            # Standard PEFT LoRA attach
            logger.info("Setting up LoRA configuration...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            logger.info("Applying LoRA to model...")
            self.base_model = get_peft_model(self.base_model, lora_config)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set pad_token_id for proper padding
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        logger.info("Backbone + LoRA ready (unsloth=%s)" % use_unsloth)
        # Optionally freeze non-LoRA params to avoid Unsloth promoting grads across the full backbone
        if self.config.freeze_non_lora:
            logger.info("Freezing non-LoRA parameters...")
            frozen = 0; trainable = 0
            for n, p in self.base_model.named_parameters():
                if 'lora_' in n or n.startswith('centroid_projection'):
                    p.requires_grad = True; trainable += p.numel()
                else:
                    p.requires_grad = False; frozen += p.numel()
            logger.info(f"Parameter freezing applied: frozen={frozen:,} trainable={trainable:,} ({trainable/(frozen+trainable+1e-9):.2%})")
        
        # Try to suppress verbose Unsloth prints if requested
        if self.config.suppress_unsloth_warnings:
            try:
                import warnings
                warnings.filterwarnings("ignore", module="unsloth")
            except Exception:
                pass

    def _align_lora_dtypes(self):
        """Cast LoRA adapter layers (lora_A/lora_B) to the base model's primary dtype.

        Prevents runtime errors like: self and mat2 must have the same dtype (BFloat16 vs Float)
        when LoRA weights remain in float32 while the quantized / dequantized base output is bf16.
        Safe because LoRA fine-tuning commonly proceeds in bf16 when the rest of the model is bf16.
        """
        try:
            target_dtype = next(self.base_model.parameters()).dtype
        except StopIteration:
            return
        changed = 0
        for module in self.base_model.modules():
            # PEFT LoRA layers expose dicts lora_A / lora_B mapping adapter name -> nn.Module
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                for key in getattr(module, 'lora_A').keys():  # adapter names
                    A = module.lora_A[key]
                    B = module.lora_B[key]
                    if hasattr(A, 'weight') and A.weight.dtype != target_dtype:
                        A.weight.data = A.weight.data.to(target_dtype)
                        changed += 1
                    if hasattr(B, 'weight') and B.weight.dtype != target_dtype:
                        B.weight.data = B.weight.data.to(target_dtype)
                        changed += 1
        if changed:
            logger.info(f"Aligned {changed} LoRA adapter weight tensors to dtype {target_dtype}.")
        # Also align centroid projection layer if mismatch
        if hasattr(self, 'centroid_projection') and self.centroid_projection.weight.dtype != target_dtype:
            self.centroid_projection.to(target_dtype)
            logger.info("Centroid projection layer cast to match base model dtype.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Forward pass through the model"""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Generate text using the model"""
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def save_model(self, path: str):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapter
        self.base_model.save_pretrained(os.path.join(path, "lora_adapter"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        
        # Save config
        with open(os.path.join(path, "config.json"), 'w') as f:
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith('_') and k != 'lora_target_modules'}
            config_dict['lora_target_modules'] = self.config.lora_target_modules
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, config: Optional[HiveLLMConfig] = None):
        """Load model from checkpoint"""
        if config is None:
            # Load config from file
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = HiveLLMConfig(**config_dict)
            else:
                config = HiveLLMConfig()
        
        model = cls(config)
        
        # Load LoRA adapter if it exists
        adapter_path = os.path.join(path, "lora_adapter")
        if os.path.exists(adapter_path):
            # Note: You'll need to implement LoRA loading logic here
            logger.info(f"LoRA adapter found at {adapter_path}")
        
        logger.info(f"Model loaded from {path}")
        return model

    # ------------------------------
    # Cluster token related methods
    # ------------------------------
    def try_add_cluster_tokens(
        self,
        board_centroids_path: Optional[str],
        move_centroids_path: Optional[str],
        initialize_embeddings: bool = True,
    ):
        """Load centroids and add corresponding tokens (if not already added)."""
        added_any = False
        if board_centroids_path and not self.board_centroids:
            self.board_centroids = _load_centroids(board_centroids_path)
            if self.board_centroids is not None:
                self.board_cluster_tokens = [f"<{self.config.board_cluster_token_prefix}_{i}>" for i in range(self.board_centroids.size(0))]
                added_any = True
        if move_centroids_path and not self.move_centroids:
            self.move_centroids = _load_centroids(move_centroids_path)
            if self.move_centroids is not None:
                self.move_cluster_tokens = [f"<{self.config.move_cluster_token_prefix}_{i}>" for i in range(self.move_centroids.size(0))]
                added_any = True
        if not added_any:
            return
        new_tokens = self.board_cluster_tokens + self.move_cluster_tokens
        # Filter tokens already present
        existing_vocab = set(self.tokenizer.get_vocab().keys())
        new_tokens = [t for t in new_tokens if t not in existing_vocab]
        if not new_tokens:
            return
        self.tokenizer.add_tokens(new_tokens, special_tokens=False)
        old_emb = self.base_model.get_input_embeddings().weight.data.clone()
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        emb_layer = self.base_model.get_input_embeddings()
        with torch.no_grad():
            emb_layer.weight.data[:old_emb.size(0)] = old_emb
        # Build id maps
        for t in self.board_cluster_tokens:
            if t in self.tokenizer.get_vocab():
                self.board_token_to_id[t] = self.tokenizer.convert_tokens_to_ids(t)
        for t in self.move_cluster_tokens:
            if t in self.tokenizer.get_vocab():
                self.move_token_to_id[t] = self.tokenizer.convert_tokens_to_ids(t)
        self.cluster_token_ids = list(self.board_token_to_id.values()) + list(self.move_token_to_id.values())
        if initialize_embeddings:
            self._initialize_cluster_token_embeddings()
        logger.info(f"Added {len(new_tokens)} cluster tokens (boards={len(self.board_cluster_tokens)}, moves={len(self.move_cluster_tokens)})")

    def _initialize_cluster_token_embeddings(self):
        """Initialize new token embeddings using centroid projection (GIN -> LLM space)."""
        emb_layer = self.base_model.get_input_embeddings()
        with torch.no_grad():
            if self.board_centroids is not None:
                proj = self.centroid_projection(self.board_centroids.to(emb_layer.weight.device))  # [B, d]
                for i, tok in enumerate(self.board_cluster_tokens):
                    if tok in self.board_token_to_id:
                        emb_layer.weight[self.board_token_to_id[tok]] = proj[i]
            if self.move_centroids is not None:
                proj = self.centroid_projection(self.move_centroids.to(emb_layer.weight.device))  # [M, d]
                for i, tok in enumerate(self.move_cluster_tokens):
                    if tok in self.move_token_to_id:
                        emb_layer.weight[self.move_token_to_id[tok]] = proj[i]
        logger.info("Initialized cluster token embeddings from centroids")

    # ------------------------------
    # Training utilities
    # ------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Weighted cross-entropy over board + move cluster tokens.

        Standard causal LM next-token prediction: we shift logits/labels and compute
        cross-entropy. If config.loss_on_cluster_tokens_only == True, restrict the
        loss to positions where the ground-truth next token is a cluster token.
        Separate board and move losses are weighted and aggregated.
        """
        cfg = self.config
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # Forward pass (no labels: we handle loss manually)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # [B,T,V]
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        vocab = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab)
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1).bool()

        # Cache sets for quick membership testing
        if not hasattr(self, '_board_id_set'):
            self._board_id_set = set(self.board_token_to_id.values())
        if not hasattr(self, '_move_id_set'):
            self._move_id_set = set(self.move_token_to_id.values())

        # Build boolean masks for cluster token types
        # Optimize: avoid converting to list and back to tensor
        label_list = flat_labels.cpu().numpy() if flat_labels.is_cuda else flat_labels.numpy()
        board_ids_np = np.array(list(self._board_id_set))
        move_ids_np = np.array(list(self._move_id_set))
        
        is_board = torch.from_numpy(np.isin(label_list, board_ids_np)).to(flat_labels.device)
        is_move = torch.from_numpy(np.isin(label_list, move_ids_np)).to(flat_labels.device)

        valid = flat_mask
        board_mask = valid & is_board
        move_mask = valid & is_move

        losses = []
        total_weight = 0.0
        metrics: Dict[str, float] = {}

        if cfg.loss_on_cluster_tokens_only:
            if board_mask.any():
                b_loss = F.cross_entropy(flat_logits[board_mask], flat_labels[board_mask])
                losses.append(cfg.board_loss_weight * b_loss)
                total_weight += cfg.board_loss_weight
                with torch.no_grad():
                    b_pred = flat_logits[board_mask].argmax(-1)
                    metrics['board_loss'] = b_loss.item()
                    metrics['board_acc'] = (b_pred == flat_labels[board_mask]).float().mean().item()
                    metrics['board_count'] = float(board_mask.sum().item())
            else:
                metrics['board_loss'] = 0.0; metrics['board_acc'] = 0.0; metrics['board_count'] = 0.0
            if move_mask.any():
                m_loss = F.cross_entropy(flat_logits[move_mask], flat_labels[move_mask])
                losses.append(cfg.move_loss_weight * m_loss)
                total_weight += cfg.move_loss_weight
                with torch.no_grad():
                    m_pred = flat_logits[move_mask].argmax(-1)
                    metrics['move_loss'] = m_loss.item()
                    metrics['move_acc'] = (m_pred == flat_labels[move_mask]).float().mean().item()
                    metrics['move_count'] = float(move_mask.sum().item())
            else:
                metrics['move_loss'] = 0.0; metrics['move_acc'] = 0.0; metrics['move_count'] = 0.0
            if not losses:  # Fallback if no cluster tokens present this batch
                std_loss = F.cross_entropy(flat_logits[valid], flat_labels[valid])
                losses.append(std_loss); total_weight = 1.0
        else:
            std_loss = F.cross_entropy(flat_logits[valid], flat_labels[valid])
            losses.append(std_loss); total_weight = 1.0

        loss = sum(losses) / max(total_weight, 1e-8)
        metrics['loss'] = loss.item()
        with torch.no_grad():
            metrics['ppl'] = math.exp(metrics['loss']) if metrics['loss'] < 50 else float('inf')
        return loss, metrics

    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.eval()
        agg = {"steps":0, "loss":0.0, "board_loss":0.0, "move_loss":0.0, "board_correct":0.0, "move_correct":0.0, "board_count":0.0, "move_count":0.0}
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches: break
                batch = {k: v.to(next(self.parameters()).device) for k, v in batch.items()}
                loss, metrics = self.training_step(batch)
                agg['steps'] += 1
                agg['loss'] += metrics['loss']
                agg['board_loss'] += metrics.get('board_loss', 0.0)
                agg['move_loss'] += metrics.get('move_loss', 0.0)
                bc = metrics.get('board_count',0.0); mc = metrics.get('move_count',0.0)
                agg['board_correct'] += metrics.get('board_acc',0.0) * bc
                agg['move_correct'] += metrics.get('move_acc',0.0) * mc
                agg['board_count'] += bc; agg['move_count'] += mc
        if agg['steps']==0:
            return {"val_loss":0.0, "val_ppl": float('inf')}
        avg_loss = agg['loss']/agg['steps']
        out = {"val_loss": avg_loss, "val_ppl": math.exp(avg_loss) if avg_loss < 50 else float('inf')}
        if agg['board_count']>0:
            out['val_board_acc'] = agg['board_correct']/agg['board_count']
        if agg['move_count']>0:
            out['val_move_acc'] = agg['move_correct']/agg['move_count']
        return out

    def fit(self, train_dl: DataLoader, val_dl: Optional[DataLoader], config: HiveLLMConfig):
        device = next(self.parameters()).device
        logger.info(f"Training device: {device}")
        logger.info(f"Training for {config.epochs} epochs with {len(train_dl)} batches per epoch")
        
        # Setup optimizer with better parameter filtering
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        total_steps = (len(train_dl) * config.epochs) // config.grad_accumulation_steps
        logger.info(f"Total training steps: {total_steps}")
        
        def lr_lambda(step):
            if step < config.warmup_steps:
                return float(step) / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, (total_steps - config.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            if config.lr_scheduler == 'linear':
                return 1.0 - progress
            # cosine
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        global_step = 0
        # Decide AMP + GradScaler usage.
        # GradScaler is ONLY needed/valid for float16. It does NOT support bfloat16 grads here (raises the NotImplementedError you saw).
        primary_dtype = next(self.parameters()).dtype
        use_cuda = torch.cuda.is_available()
        use_fp16 = use_cuda and primary_dtype == torch.float16
        use_bf16 = use_cuda and primary_dtype == torch.bfloat16
        # Autocast: enable for fp16 or bf16, but GradScaler only for fp16.
        if use_fp16:
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            scaler = None  # No scaling for bf16 or full precision
        if use_bf16:
            logger.info("[fit] Detected bfloat16 parameters; disabling GradScaler (not needed) and using bf16 autocast.")
        elif use_fp16:
            logger.info("[fit] Using fp16 autocast with GradScaler.")
        else:
            logger.info("[fit] Training in full precision (no autocast / scaler).")
        # Safety: if torch.compile not globally disabled, try disabling just-in-time here.
        if os.name == 'nt' and os.environ.get("TORCH_COMPILE_DISABLE", "").lower() not in ("1", "true"):
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            logger.info("[fit] Disabling torch.compile during training to avoid Windows cl.exe requirement.")
            
        # Clear CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"CUDA memory before training: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
        
        self.train()
        
        for epoch in range(config.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
            
            # Create progress bar if tqdm available
            if _HAS_TQDM:
                pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
                data_iter = enumerate(pbar)
            else:
                data_iter = enumerate(train_dl)
                
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, batch in data_iter:
                # Move to device with non_blocking for better GPU utilization
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                if use_fp16 or use_bf16:
                    # Use new torch.amp.autocast API (device_type='cuda').
                    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
                        loss, metrics = self.training_step(batch)
                        loss = loss / config.grad_accumulation_steps
                else:
                    loss, metrics = self.training_step(batch)
                    loss = loss / config.grad_accumulation_steps

                # Backward + optimizer step
                if scaler is not None:  # fp16 path
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (batch_idx + 1) % config.grad_accumulation_steps == 0:
                    if scaler is not None:
                        # Unscale & clip
                        try:
                            scaler.unscale_(optim)
                        except NotImplementedError:
                            # Fallback (should not happen for fp16, but guard anyway)
                            logger.warning("GradScaler unscale_ not implemented for current dtype; performing direct step.")
                        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
                        scaler.step(optim)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
                        optim.step()
                    optim.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    
                    # Update progress bar if available
                    if _HAS_TQDM and 'pbar' in locals():
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'ppl': f"{metrics['ppl']:.2f}",
                            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                        })
                    
                    if global_step % config.log_every == 0:
                        lr = scheduler.get_last_lr()[0]
                        extras = []
                        if 'board_acc' in metrics and metrics.get('board_count',0)>0:
                            extras.append(f"b_acc {metrics['board_acc']:.3f}")
                        if 'move_acc' in metrics and metrics.get('move_count',0)>0:
                            extras.append(f"m_acc {metrics['move_acc']:.3f}")
                        if 'board_loss' in metrics and metrics.get('board_count',0)>0:
                            extras.append(f"b_loss {metrics['board_loss']:.3f}")
                        if 'move_loss' in metrics and metrics.get('move_count',0)>0:
                            extras.append(f"m_loss {metrics['move_loss']:.3f}")
                        logger.info(f"epoch {epoch} step {global_step} loss {metrics['loss']:.4f} ppl {metrics['ppl']:.2f} {' '.join(extras)} lr {lr:.2e}")
                    if val_dl and global_step % config.eval_every == 0:
                        # Clear cache before evaluation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        eval_metrics = self.evaluate(val_dl, max_batches=10)
                        extras = []
                        if 'val_board_acc' in eval_metrics:
                            extras.append(f"val_b_acc {eval_metrics['val_board_acc']:.3f}")
                        if 'val_move_acc' in eval_metrics:
                            extras.append(f"val_m_acc {eval_metrics['val_move_acc']:.3f}")
                        logger.info(f"[eval] step {global_step} val_loss {eval_metrics['val_loss']:.4f} val_ppl {eval_metrics['val_ppl']:.2f} {' '.join(extras)}")
                        self.train()  # Make sure we're back in training mode
                    if config.save_every and global_step % config.save_every == 0:
                        save_dir = os.path.join(config.output_dir, f"checkpoint_step{global_step}")
                        self.save_model(save_dir)
                        
                # Track epoch metrics
                epoch_loss += metrics['loss']
                epoch_steps += 1
                
            # End of epoch logging
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")
            
            # End epoch eval
            if val_dl:
                eval_metrics = self.evaluate(val_dl)
                extras = []
                if 'val_board_acc' in eval_metrics:
                    extras.append(f"val_b_acc {eval_metrics['val_board_acc']:.3f}")
                if 'val_move_acc' in eval_metrics:
                    extras.append(f"val_m_acc {eval_metrics['val_move_acc']:.3f}")
                logger.info(f"[epoch {epoch}] val_loss {eval_metrics['val_loss']:.4f} val_ppl {eval_metrics['val_ppl']:.2f} {' '.join(extras)}")
        # Final save
        self.save_model(config.output_dir)
        logger.info("Training complete; model saved.")


def main():
    """Train / finetune LLM on cluster token sequences."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Optional JSON config file overriding defaults')
    parser.add_argument('--train_cache', type=str, default=None)
    parser.add_argument('--val_cache', type=str, default=None)
    parser.add_argument('--board_centroids', type=str, default=None)
    parser.add_argument('--move_centroids', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--board_loss_weight', type=float, default=None)
    parser.add_argument('--move_loss_weight', type=float, default=None)
    parser.add_argument('--loss_on_cluster_tokens_only', action='store_true')
    parser.add_argument('--all_tokens_loss', action='store_true', help='If set, compute loss over all tokens (overrides --loss_on_cluster_tokens_only).')

    args = parser.parse_args()

    config = HiveLLMConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            override = json.load(f)
        for k, v in override.items():
            if hasattr(config, k):
                setattr(config, k, v)
    # CLI overrides
    if args.train_cache: config.train_cache_path = args.train_cache
    if args.val_cache: config.val_cache_path = args.val_cache
    if args.board_centroids: config.board_centroids_path = args.board_centroids
    if args.move_centroids: config.move_centroids_path = args.move_centroids
    if args.epochs: config.epochs = args.epochs
    if args.lr: config.learning_rate = args.lr
    if args.batch_size: config.batch_size = args.batch_size
    if args.output_dir: config.output_dir = args.output_dir
    if args.board_loss_weight is not None: config.board_loss_weight = args.board_loss_weight
    if args.move_loss_weight is not None: config.move_loss_weight = args.move_loss_weight
    if args.loss_on_cluster_tokens_only: config.loss_on_cluster_tokens_only = True
    if args.all_tokens_loss:
        config.loss_on_cluster_tokens_only = False
    os.makedirs(config.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    logger.info(f"Set random seed to {config.seed}")

    logger.info("Creating model...")
    model = HiveLLMModel(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Moving model to device: {device}")
    model.to(device)

    # Load dataset caches
    logger.info("Loading dataset caches...")
    def _load_cache(path):
        if not path or not os.path.exists(path):
            return []
        logger.info(f"Loading cache from: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    train_samples = _load_cache(config.train_cache_path)
    val_samples = _load_cache(config.val_cache_path)
    if not train_samples:
        logger.error("No training samples loaded. Provide --train_cache pointing to train_sequential_cache.pkl")
        logger.error(f"Attempted to load from: {config.train_cache_path}")
        if config.train_cache_path and os.path.exists(config.train_cache_path):
            logger.error("File exists but contains no valid samples")
        else:
            logger.error("File does not exist")
        return
    logger.info(f"Loaded {len(train_samples)} train samples; {len(val_samples)} val samples")
    
    # Log a sample to debug data format
    if train_samples:
        sample = train_samples[0]
        logger.info(f"Sample keys: {list(sample.keys())}")
        if 'board_cluster_ids_sequence' in sample:
            board_seq = sample['board_cluster_ids_sequence']
            logger.info(f"Board sequence type: {type(board_seq)}, length: {len(board_seq) if hasattr(board_seq, '__len__') else 'N/A'}")
        if 'chosen_move_cluster_ids_sequence' in sample:
            move_seq = sample['chosen_move_cluster_ids_sequence']
            logger.info(f"Move sequence type: {type(move_seq)}, length: {len(move_seq) if hasattr(move_seq, '__len__') else 'N/A'}")

    logger.info("Creating datasets...")
    dataset_train = GameClusterSequenceDataset(
        train_samples,
        model.board_cluster_tokens,
        model.move_cluster_tokens,
        tokenizer=model.tokenizer,
        add_eos=config.add_eos_token,
    )
    dataset_val = GameClusterSequenceDataset(
        val_samples,
        model.board_cluster_tokens,
        model.move_cluster_tokens,
        tokenizer=model.tokenizer,
        add_eos=config.add_eos_token,
    ) if val_samples else None

    logger.info("Creating data loaders...")
    # Optimize DataLoader for better GPU utilization
    # Note: On Windows, multiprocessing with transformers tokenizers can cause issues
    # So we use num_workers=0 on Windows to avoid pickle/spawn errors
    use_multiprocessing = torch.cuda.is_available() and os.name != 'nt'
    dataloader_kwargs = {
        'batch_size': config.batch_size,
        'collate_fn': GameClusterSequenceDataset.collate,
        'pin_memory': torch.cuda.is_available(),
        'num_workers': 2 if use_multiprocessing else 0,
    }
    
    if not use_multiprocessing:
        logger.info("Using single-threaded DataLoader (Windows + tokenizer compatibility)")
    
    train_dl = DataLoader(dataset_train, shuffle=True, **dataloader_kwargs)
    val_dl = DataLoader(dataset_val, shuffle=False, **dataloader_kwargs) if dataset_val else None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params total={total_params:,} trainable={trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    model.fit(train_dl, val_dl, config)


if __name__ == "__main__":
    main()
