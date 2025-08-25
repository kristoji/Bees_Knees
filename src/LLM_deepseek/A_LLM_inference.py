#!/usr/bin/env python3
"""
Inference script for sequential HIVE LLM gameplay.
Load trained model and play with historical context.
"""

from pyexpat import model
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque

from A_LLM_trainer import SequentialHiveLLMPlayer, SequentialHiveLLMConfig  # or make it a proper module name
from improved_projection_layers import HiveProjectionModule
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


class SequentialHiveLLMInference:
    """Inference engine for sequential HIVE LLM gameplay with historical context"""
    
    def __init__(self, model_path: str, sequence_length: int = 6, device: str = "auto"):
        """
        Initialize sequential inference engine
        
        Args:
            model_path: Path to saved model directory
            sequence_length: Length of historical context to maintain
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.model_path = Path(model_path)
        self.sequence_length = sequence_length
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize game history buffers
        self.board_history = deque(maxlen=sequence_length)
        self.move_history = deque(maxlen=sequence_length)
        
        logger.info("Sequential model loaded successfully")
    
    def _load_config(self) -> SequentialHiveLLMConfig:
        """Load model configuration"""
        config_path = self.model_path / "config.json"
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return SequentialHiveLLMConfig(**config_dict)
    
    def _load_model(self) -> SequentialHiveLLMPlayer:
        """Load the trained sequential model"""
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        lora_path = self.model_path / "lora_adapter"
        if lora_path.exists():
            base_model = PeftModel.from_pretrained(base_model, str(lora_path))
            logger.info("Loaded LoRA adapter")
        
        # Load tokenizer
        tokenizer_path = self.model_path / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True
            )
        
        # Create model wrapper
        model = SequentialHiveLLMPlayer(self.config)
        model.base_model = base_model
        model.tokenizer = tokenizer
        # Ensure embedding table matches tokenizer (incl. your added special tokens)
        model.base_model.resize_token_embeddings(len(model.tokenizer))

        # Also ensure pad token exists and != eos
        if model.tokenizer.pad_token is None or model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
            model.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.base_model.resize_token_embeddings(len(model.tokenizer))
        
    # Projection bridge (AE) is fixed and loaded inside the player; no separate state to load.
        
        model.to(self.device)
        return model
    
    def reset_game_history(self):
        """Reset the game history buffers for a new game"""
        self.board_history.clear()
        self.move_history.clear()
        logger.info("Game history reset")
    
    def add_to_history(self, board_embedding: np.ndarray, move_embedding: np.ndarray):
        """
        Add a board state and move to the historical context
        
        Args:
            board_embedding: Board state embedding [1, gin_dim]
            move_embedding: Move embedding [1, move_dim]
        """
        # Convert to tensors and add to history
        # Keep board entries as [1, gin_dim] so that stacking yields [seq_len, 1, gin_dim]
        board_tensor = torch.as_tensor(board_embedding, dtype=torch.float32)
        if board_tensor.dim() == 1:
            board_tensor = board_tensor.unsqueeze(0)
        # Moves are raw vectors [move_dim]; store as [1, move_dim]
        move_tensor = torch.as_tensor(move_embedding, dtype=torch.float32)
        if move_tensor.dim() == 1:
            move_tensor = move_tensor.unsqueeze(0)

        self.board_history.append(board_tensor)
        self.move_history.append(move_tensor)

        logger.debug(f"Added to history. Current history length: {len(self.board_history)}")
    
    def predict_move(self, 
                    current_board_embedding: np.ndarray,
                    legal_moves: List[np.ndarray],
                    temperature: float = 0.1,
                    top_k: int = 5) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Predict the best move given current board state and historical context.
        The model generates freely without seeing legal moves, then we find the closest legal move.
        
        Args:
            current_board_embedding: Current board state embedding [1, gin_dim]
            legal_moves: List of legal move embeddings, each [move_dim] (for similarity matching only)
            temperature: Sampling temperature (not used - model generates freely)
            top_k: Number of top moves to consider (not used - model generates freely)
            
        Returns:
            move_index: Index of chosen legal move (closest to generated move)
            confidence: Confidence score (similarity to closest legal move)
            predicted_move_embedding: Predicted move in LLM space
            predicted_next_state: Predicted next board state in LLM space
        """
        # Prepare context tensors
        if len(self.board_history) == 0:
            # No history available, create dummy context
            dummy_board = torch.zeros(1, 1, self.config.gin_embedding_dim)
            dummy_move = torch.zeros(1, 1, self.config.gin_embedding_dim)
            
            context_boards = dummy_board.repeat(self.sequence_length, 1, 1).unsqueeze(0)
            context_moves = dummy_move.repeat(self.sequence_length, 1, 1).unsqueeze(0)
        else:
            # Pad history to sequence length if needed
            actual_length = len(self.board_history)
            
            if actual_length < self.sequence_length:
                # Pad with zeros
                padding_needed = self.sequence_length - actual_length
                
                zero_board = torch.zeros_like(self.board_history[0])
                zero_move = torch.zeros_like(self.move_history[0])
                
                padded_boards = [zero_board] * padding_needed + list(self.board_history)
                padded_moves = [zero_move] * padding_needed + list(self.move_history)
                
                context_boards = torch.stack(padded_boards).unsqueeze(0)  # [1, seq_len, 1, gin_dim]
                context_moves = torch.stack(padded_moves).unsqueeze(0)    # [1, seq_len, 1, move_dim]
            else:
                # Use recent history
                context_boards = torch.stack(list(self.board_history)).unsqueeze(0)
                context_moves = torch.stack(list(self.move_history)).unsqueeze(0)
        
        # Prepare current board tensor: ensure shape [1, 1, gin_dim]
        cbt = torch.as_tensor(current_board_embedding, dtype=torch.float32)
        if cbt.dim() == 1:
            cbt = cbt.unsqueeze(0)
        current_board_tensor = cbt.unsqueeze(0)
        
        # Prepare legal moves tensor
        max_moves = len(legal_moves)
        move_dim = legal_moves[0].shape[0]
        
        legal_move_embeddings = np.zeros((1, max_moves, 1, move_dim))
        legal_move_mask = np.zeros((1, max_moves), dtype=bool)
        
        for i, move in enumerate(legal_moves):
            legal_move_embeddings[0, i, 0] = move
            legal_move_mask[0, i] = True
        
        legal_move_tensor = torch.FloatTensor(legal_move_embeddings).to(self.device)
        legal_mask_tensor = torch.BoolTensor(legal_move_mask).to(self.device)
        
        # Move tensors to device
        context_boards = context_boards.to(self.device)
        context_moves = context_moves.to(self.device)
        current_board_tensor = current_board_tensor.to(self.device)
        
        # Generate prediction (model generates freely without seeing legal moves)
        with torch.no_grad():
            move_idx, confidence, pred_move_emb, pred_next_state = self.model.generate_move(
                context_boards, context_moves, current_board_tensor, 
                legal_move_tensor, legal_mask_tensor  # Still passed for similarity matching
            )
        
        # Convert predictions back to numpy
        predicted_move_embedding = pred_move_emb.cpu().numpy()
        predicted_next_state = pred_next_state.cpu().numpy()
        
        return move_idx, confidence, predicted_move_embedding, predicted_next_state
    
    def play_sequential_game(self, 
                           initial_board: np.ndarray,
                           move_generator_fn,
                           board_updater_fn,
                           max_moves: int = 100,
                           verbose: bool = True) -> Dict:
        """
        Play a complete game with sequential context
        
        Args:
            initial_board: Initial board state embedding
            move_generator_fn: Function that generates legal moves given board state
            board_updater_fn: Function that updates board state given a move
            max_moves: Maximum number of moves
            verbose: Whether to print game progress
            
        Returns:
            Game result dictionary
        """
        # Reset history for new game
        self.reset_game_history()
        
        game_history = []
        current_board = initial_board.copy()
        
        for move_num in range(max_moves):
            # Generate legal moves
            legal_moves = move_generator_fn(current_board)
            
            if not legal_moves:
                if verbose:
                    print("No legal moves available - game over")
                break
            
            # Predict move using sequential context
            move_idx, confidence, pred_move_emb, pred_next_state = self.predict_move(
                current_board, legal_moves
            )
            
            chosen_move = legal_moves[move_idx]
            
            if verbose:
                history_len = len(self.board_history)
                print(f"Move {move_num + 1}: Model generated freely, chose closest legal move {move_idx} "
                      f"(similarity: {confidence:.3f}, context: {history_len} moves)")
            
            # Record move
            game_history.append({
                'move_number': move_num + 1,
                'board_state': current_board.copy(),
                'legal_moves': legal_moves.copy(),
                'chosen_move_index': move_idx,
                'chosen_move': chosen_move.copy(),
                'confidence': confidence,
                'predicted_move_embedding': pred_move_emb,
                'predicted_next_state': pred_next_state,
                'context_length': len(self.board_history)
            })
            
            # Add current state and chosen move to history
            self.add_to_history(current_board, chosen_move)
            
            # Update board state
            next_board = board_updater_fn(current_board, chosen_move)
            current_board = next_board
            
            # Check for game end conditions
            if self._is_game_over(current_board):
                if verbose:
                    print(f"Game over after {move_num + 1} moves")
                break
        
        return {
            'game_history': game_history,
            'final_board': current_board,
            'total_moves': len(game_history),
            'final_context_length': len(self.board_history)
        }
    
    def _is_game_over(self, board_state: np.ndarray) -> bool:
        """
        Check if game is over (placeholder - implement based on your game rules)
        """
        # This is a placeholder - you'll need to implement game-over detection
        return False
    
    def benchmark_sequential_model(self, 
                                 test_data_path: str,
                                 num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark sequential model performance on test data
        
        Args:
            test_data_path: Path to test dataset with sequential samples
            num_samples: Number of samples to evaluate
            
        Returns:
            Performance metrics
        """
        # Load test/validation data (sequential format)
        root = Path(test_data_path)
        # Prefer test cache if present; else fall back to validation, then train
        candidates = [
            root / "test_sequential_cache.pkl",
            root / "validation_sequential_cache.pkl",
            root / "train_sequential_cache.pkl",
        ]
        test_cache_path = next((p for p in candidates if p.exists()), None)
        if test_cache_path is None:
            raise FileNotFoundError(f"No sequential cache found in {root} (looked for test/validation/train caches)")

        import pickle
        with open(test_cache_path, 'rb') as f:
            test_samples = pickle.load(f)
        
        # Evaluate samples
        correct_predictions = 0
        total_confidence = 0
        context_length_sum = 0
        
        num_samples = min(num_samples, len(test_samples))
        
        for i in range(num_samples):
            sample = test_samples[i]
            
            # Extract sample data
            context_boards = sample['context_board_embeddings']  # [seq_len, 1, gin_dim]
            context_moves = sample['context_move_embeddings']    # [seq_len, 1, move_dim]
            current_board = sample['current_board_embedding']    # [1, gin_dim]
            legal_moves = sample['legal_move_embeddings']        # [num_moves, 1, move_dim]
            target_move_idx = sample['chosen_move_idx']
            
            # Manually set history (for evaluation)
            self.board_history.clear()
            self.move_history.clear()
            
            seq_len = context_boards.shape[0]
            for j in range(seq_len):
                self.board_history.append(context_boards[j:j+1])  # [1, 1, gin_dim]
                self.move_history.append(context_moves[j:j+1])    # [1, 1, move_dim]
            
            # Extract legal moves as list
            num_legal_moves = legal_moves.shape[0]
            legal_moves_list = [legal_moves[j, 0].numpy() for j in range(num_legal_moves)]
            
            if not legal_moves_list:
                continue
            
            # Predict using sequential context (free generation)
            pred_idx, confidence, _, _ = self.predict_move(
                current_board.numpy(), legal_moves_list
            )
            
            # Check if prediction matches target
            if pred_idx == target_move_idx:
                correct_predictions += 1
            
            total_confidence += confidence
            context_length_sum += len(self.board_history)
        
        accuracy = correct_predictions / num_samples
        avg_confidence = total_confidence / num_samples
        avg_context_length = context_length_sum / num_samples
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'average_context_length': avg_context_length,
            'samples_evaluated': num_samples
        }
    
    def get_model_analysis(self, 
                          board_embedding: np.ndarray,
                          legal_moves: List[np.ndarray]) -> Dict:
        """
        Get detailed analysis of model predictions (free generation with similarity matching)
        
        Args:
            board_embedding: Current board state
            legal_moves: List of legal move embeddings (for similarity analysis)
            
        Returns:
            Analysis dictionary with similarities for all moves
        """
        # Get prediction (model generates freely)
        pred_idx, confidence, pred_move_emb, pred_next_state = self.predict_move(
            board_embedding, legal_moves
        )
        
        # Calculate similarities with all legal moves
        move_similarities = []
        
        # Prepare tensors for similarity calculation
        legal_move_embeddings = np.zeros((1, len(legal_moves), 1, legal_moves[0].shape[0]))
        legal_move_mask = np.zeros((1, len(legal_moves)), dtype=bool)
        
        for i, move in enumerate(legal_moves):
            legal_move_embeddings[0, i, 0] = move
            legal_move_mask[0, i] = True
        
        legal_move_tensor = torch.FloatTensor(legal_move_embeddings).to(self.device)
        legal_mask_tensor = torch.BoolTensor(legal_move_mask).to(self.device)
        
        # Compare in GIN space: decode predicted LLM move and compare to raw legal GIN embeddings
        with torch.no_grad():
            pred_move_tensor_llm = torch.FloatTensor(pred_move_emb).to(self.device)
            pred_move_gin = self.model.projection_module.move_projection_inverse(pred_move_tensor_llm)  # [gin_dim]
            legal_moves_gin = legal_move_tensor.squeeze(0).squeeze(1)  # [num_moves, gin_dim]

            similarities = F.cosine_similarity(
                pred_move_gin.unsqueeze(0),
                legal_moves_gin,
                dim=1
            ).cpu().numpy()
        
        # Create analysis
        for i, (move, sim) in enumerate(zip(legal_moves, similarities)):
            move_similarities.append({
                'move_index': i,
                'similarity': float(sim),
                'is_chosen': i == pred_idx,
                'move_embedding': move.tolist()
            })
        
        # Sort by similarity
        move_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'predicted_move_index': pred_idx,
            'confidence': confidence,
            'context_length': len(self.board_history),
            'move_similarities': move_similarities,
            'predicted_move_embedding': pred_move_emb.tolist(),
            'predicted_next_state': pred_next_state.tolist(),
            'generation_mode': 'free_generation_with_similarity_matching'
        }


def main():
    """Main inference script for sequential HIVE LLM"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequential HIVE LLM Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained sequential model directory")
    parser.add_argument("--mode", type=str, choices=["play", "benchmark", "analyze"], 
                       default="benchmark", help="Mode: play, benchmark, or analyze")
    parser.add_argument("--test_data", type=str, default=None,
                       help="Path to test data for benchmarking")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--sequence_length", type=int, default=6,
                       help="Length of historical context sequence")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for benchmarking")
    
    args = parser.parse_args()
    
    # Initialize sequential inference engine
    engine = SequentialHiveLLMInference(
        args.model_path, 
        sequence_length=args.sequence_length,
        device=args.device
    )
    
    if args.mode == "benchmark":
        if not args.test_data:
            raise ValueError("Test data path required for benchmarking")
        
        print("Benchmarking sequential model...")
        results = engine.benchmark_sequential_model(args.test_data, args.num_samples)
        
        print("\n=== Sequential Benchmark Results ===")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Average Confidence: {results['average_confidence']:.3f}")
        print(f"Average Context Length: {results['average_context_length']:.1f}")
        print(f"Samples Evaluated: {results['samples_evaluated']}")
    
    elif args.mode == "analyze":
        print("Analysis mode - demonstrating model analysis")
        
        # Create dummy data for demonstration
        dummy_board = np.random.randn(1, 256)  # Assuming gin_dim = 256
        dummy_moves = [np.random.randn(256) for _ in range(5)]  # 5 legal moves
        
        # Add some dummy history
        for i in range(3):
            dummy_hist_board = np.random.randn(1, 256)
            dummy_hist_move = np.random.randn(256)
            engine.add_to_history(dummy_hist_board, dummy_hist_move)
        
        # Get analysis
        analysis = engine.get_model_analysis(dummy_board, dummy_moves)
        
        print("\n=== Model Analysis (Free Generation) ===")
        print(f"Predicted Move Index: {analysis['predicted_move_index']}")
        print(f"Similarity Score: {analysis['confidence']:.3f}")
        print(f"Context Length: {analysis['context_length']}")
        print(f"Generation Mode: {analysis['generation_mode']}")
        print("\nMove Similarities (Legal moves ranked by similarity to generated move):")
        for i, move_info in enumerate(analysis['move_similarities'][:3]):  # Show top 3
            chosen_mark = " ⭐" if move_info['is_chosen'] else ""
            print(f"  {i+1}. Move {move_info['move_index']}: {move_info['similarity']:.3f}{chosen_mark}")
    
    elif args.mode == "play":
        print("Interactive play mode - Free Generation")
        print("Sequential model ready (generates moves freely without legal move constraints)")
        print("You'll need to integrate with your HIVE game engine")
        
        # Example of how to use the sequential engine
        def dummy_move_generator(board_state):
            """Generate dummy legal moves for demonstration"""
            return [np.random.randn(256) for _ in range(np.random.randint(3, 8))]
        
        def dummy_board_updater(board_state, move):
            """Update board state (dummy implementation)"""
            return np.random.randn(*board_state.shape)
        
        # Example game
        initial_board = np.random.randn(1, 256)
        game_result = engine.play_sequential_game(
            initial_board, 
            dummy_move_generator,
            dummy_board_updater,
            max_moves=10,
            verbose=True
        )
        
        print(f"\nDemo game completed (Free Generation Mode):")
        print(f"Total moves: {game_result['total_moves']}")
        print(f"Final context length: {game_result['final_context_length']}")
        print("Note: Model generated moves freely, then selected closest legal moves")


if __name__ == "__main__":
    main()