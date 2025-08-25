#!/usr/bin/env python3
"""
Simplified HIVE LLM architecture with model loading utilities.
Contains only the essential components for LLM and GNN model loading.
"""
import torch
import torch.nn as nn
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from torch import bfloat16

# Hugging Face and LoRA imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HiveLLMConfig:
    """Configuration for HIVE LLM"""
    
    # Model configuration
    base_model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    gin_embedding_dim: int = 256
    llm_token_dim: int = 4096
    
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
    """Main HIVE LLM model with essential components"""
    
    def __init__(self, config: HiveLLMConfig):
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

        # Choose safe compute dtype
        self.compute_dtype = bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        # Initialize quantization config
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=config.use_nested_quant,
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

        # Configure model for stability
        self.base_model.config.use_cache = config.use_cache
        if hasattr(self.base_model, "generation_config"):
            self.base_model.generation_config.use_cache = config.use_cache

        # Resize embeddings after tokenizer changes
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Add special tokens
        self._add_special_tokens()

        # Prepare model for k-bit training if using quantization
        if config.use_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # Initialize LoRA
        self._setup_lora()

        # Load GNN model
        self.gnn_loader = GNNLoader()
        self.gnn_model = self.gnn_loader.load_gnn_model(config.gnn_model_path)
        
        logger.info("HiveLLM model initialized successfully")
    
    def _add_special_tokens(self):
        """Add special tokens for HIVE game representation"""
        special_tokens = ["<BOARD>", "<MOVE>", "<CHOSEN_MOVE>", "<NEXT_STATE>"]
        add = {"additional_special_tokens": special_tokens}

        # Add a real pad token if missing
        if self.tokenizer.pad_token is None:
            add["pad_token"] = "<|pad|>"

        self.tokenizer.add_special_tokens(add)

        # Store token IDs for reference
        self.board_token_id = self.tokenizer.convert_tokens_to_ids("<BOARD>")
        self.move_token_id = self.tokenizer.convert_tokens_to_ids("<MOVE>")
        self.chosen_move_token_id = self.tokenizer.convert_tokens_to_ids("<CHOSEN_MOVE>")
        self.next_state_token_id = self.tokenizer.convert_tokens_to_ids("<NEXT_STATE>")

        # Resize embedding table to match tokenizer
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
    def _setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        logger.info("LoRA configuration applied successfully")

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


def main():
    """Example usage of the simplified HIVE LLM model"""
    # Configuration
    config = HiveLLMConfig(
        base_model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        gin_embedding_dim=256,
        llm_token_dim=4096,
        gnn_model_path="src/models/pretrain_GIN_3.pt",
        output_dir="models/hive_llm_simple"
    )
    
    # Initialize model
    model = HiveLLMModel(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


if __name__ == "__main__":
    main()
