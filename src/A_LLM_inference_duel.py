#!/usr/bin/env python3
"""
LLM Inference Script for Hive Game
Plays the trained LLM model against a random player using cluster tokens and JSON predictions.
"""

import os
import json
import pickle
import logging
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Game engine imports
from engine.board import Board
from engine.game import Move, Bug, Position
from engine.enums import GameState, PlayerColor
from ai.brains import Random

# GNN imports for embedding computation
from ai.oracleGNN import OracleGNN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HiveLLMInference:
    """Inference wrapper for the trained Hive LLM model"""
    
    def __init__(self, model_dir: str, gnn_model_path: str, device: str = "cuda"):
        self.device = device
        self.model_dir = model_dir
        self.gnn_model_path = gnn_model_path
        
        # Load model components
        self._load_model()
        self._load_gnn()
        self._load_cluster_mappings()
        self._load_system_prompt()
        
    def _load_model(self):
        """Load the trained LLM model and tokenizer - Unsloth compatible version"""
        logger.info(f"Loading model from {self.model_dir}")
        
        # Load config to get base model name
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            # Try parent directory
            parent_config_path = os.path.join(os.path.dirname(self.model_dir), "config.json")
            if os.path.exists(parent_config_path):
                config_path = parent_config_path
            else:
                raise FileNotFoundError(f"Config not found at {config_path} or {parent_config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        base_model_name = config.get("base_model_name", "unsloth/gemma-2-9b-it-bnb-4bit")
        logger.info(f"Base model from config: {base_model_name}")
        
        # Check if this is an Unsloth model
        is_unsloth_model = base_model_name.startswith("unsloth/")
        
        if is_unsloth_model:
            try:
                from unsloth import FastLanguageModel
                logger.info("Loading model using Unsloth FastLanguageModel...")
                
                # Load base model and tokenizer via Unsloth
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_name,
                    max_seq_length=config.get("max_seq_length", 512),
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    load_in_4bit=config.get("use_4bit", True),
                )
                
                # Load the fine-tuned adapter
                adapter_path = os.path.join(self.model_dir, "adapter")
                if not os.path.exists(adapter_path):
                    # Check if adapter files are directly in model_dir
                    if os.path.exists(os.path.join(self.model_dir, "adapter_config.json")):
                        adapter_path = self.model_dir
                    else:
                        raise FileNotFoundError(f"Adapter not found at {adapter_path}")
                
                logger.info(f"Loading adapter from: {adapter_path}")
                
                # Load adapter using Unsloth's method
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
                
                # Set to inference mode
                FastLanguageModel.for_inference(model)
                
                self.model = model
                self.tokenizer = tokenizer
                
                # Handle processor if it's Gemma3
                if hasattr(tokenizer, 'tokenizer'):
                    # This is a processor, extract the base tokenizer
                    self.processor = tokenizer
                    self.base_tokenizer = tokenizer.tokenizer
                else:
                    self.processor = None
                    self.base_tokenizer = tokenizer
                
                # Ensure pad token is set
                if self.base_tokenizer.pad_token is None:
                    self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
                    self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
                
                logger.info("Model and adapter loaded successfully via Unsloth")
                
            except ImportError:
                logger.error("Unsloth is required but not installed. Install it with: pip install unsloth")
                raise
            except Exception as e:
                logger.error(f"Failed to load Unsloth model: {e}")
                raise
                
        else:
            # Non-Unsloth model - use standard HuggingFace loading
            logger.info("Loading model using standard HuggingFace approach...")
            
            # Load tokenizer
            tokenizer_path = None
            possible_tokenizer_paths = [
                os.path.join(self.model_dir, "tokenizer"),
                os.path.join(self.model_dir, "processor"),
                self.model_dir,
                os.path.join(os.path.dirname(self.model_dir), "tokenizer"),
                os.path.join(os.path.dirname(self.model_dir), "processor")
            ]
            
            for path in possible_tokenizer_paths:
                if os.path.exists(path) and (os.path.exists(os.path.join(path, "tokenizer_config.json")) or 
                                            os.path.exists(os.path.join(path, "tokenizer.json"))):
                    tokenizer_path = path
                    break
            
            if tokenizer_path is None:
                raise FileNotFoundError(f"Tokenizer not found in any of: {possible_tokenizer_paths}")
            
            logger.info(f"Loading tokenizer from: {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load the base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            
            # Load adapter if exists
            adapter_path = os.path.join(self.model_dir, "adapter")
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                from peft import PeftModel
                logger.info(f"Loading adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
            self.model.eval()
            self.processor = None
            self.base_tokenizer = self.tokenizer
            
            logger.info("Model loaded successfully via HuggingFace")
        
    def _load_gnn(self):
        """Load GNN model for embedding computation"""
        logger.info(f"Loading GNN model from {self.gnn_model_path}")
        
        kwargs_network = {
            'conv_type': 'GIN',
            'num_layers': 6,
            'gat_heads': 4,
            'gat_concat': True,
            'conv_dropout': 0.1,
            'mlp_dropout': 0.1,
            'final_dropout': 0.2,
            'use_batch_norm': False,
            'use_layer_norm': True,
            'use_residual': False,
            'pooling': 'add',
            'mlp_layers': 3,
            'final_mlp_layers': 3
        }
        
        self.gnn_model = OracleGNN(device=str(self.device), hidden_dim=256, **kwargs_network)
        self.gnn_model.load(self.gnn_model_path)
        self.gnn_model.network.eval()
        
    def _load_cluster_mappings(self):
        """Load cluster centroids and token mappings"""
        # Look for token mapping in multiple locations
        mapping_paths = [
            os.path.join(self.model_dir, "token_centroid_mapping.json"),
            os.path.join(os.path.dirname(self.model_dir), "token_centroid_mapping.json"),  # Parent directory
            os.path.join("data", "sequential_hive_llm_dataset", "token_centroid_mapping.json"),
            "token_centroid_mapping.json"
        ]
        
        mapping_path = None
        for path in mapping_paths:
            if os.path.exists(path):
                mapping_path = path
                break
                
        if mapping_path is None:
            raise FileNotFoundError(f"Token mapping not found in any of: {mapping_paths}")
            
        logger.info(f"Loading token mapping from: {mapping_path}")
        with open(mapping_path, 'r') as f:
            self.token_mapping = json.load(f)
            
        # Load cluster centroids from multiple possible locations
        centroid_base_paths = [
            "models/clustering",
            "clustering_models", 
            os.path.join(os.path.dirname(self.model_dir), "clustering"),
            "."
        ]
        
        self.board_centroids = None
        self.move_centroids = None
        
        for base_path in centroid_base_paths:
            board_centroids_path = os.path.join(base_path, "boards", "cluster_centroids_kmeans_best.pkl")
            move_centroids_path = os.path.join(base_path, "moves", "cluster_centroids_kmeans_best.pkl")
            
            if os.path.exists(board_centroids_path) and self.board_centroids is None:
                logger.info(f"Loading board centroids from: {board_centroids_path}")
                with open(board_centroids_path, 'rb') as f:
                    board_data = pickle.load(f)
                    self.board_centroids = torch.tensor(board_data['centroids'], dtype=torch.float32)
                    
            if os.path.exists(move_centroids_path) and self.move_centroids is None:
                logger.info(f"Loading move centroids from: {move_centroids_path}")
                with open(move_centroids_path, 'rb') as f:
                    move_data = pickle.load(f)
                    self.move_centroids = torch.tensor(move_data['centroids'], dtype=torch.float32)
                    
        if self.board_centroids is None:
            logger.warning("Board centroids not found - will use dummy clustering")
        if self.move_centroids is None:
            logger.warning("Move centroids not found - will use dummy clustering")
            
        # Create reverse mappings for quick lookup
        self.board_token_to_id = {item['token']: item['index'] for item in self.token_mapping.get('board', [])}
        self.move_token_to_id = {item['token']: item['index'] for item in self.token_mapping.get('move', [])}
        self.board_id_to_token = {v: k for k, v in self.board_token_to_id.items()}
        self.move_id_to_token = {v: k for k, v in self.move_token_to_id.items()}
        
        logger.info(f"Loaded {len(self.board_token_to_id)} board tokens, {len(self.move_token_to_id)} move tokens")
        
    def _load_system_prompt(self):
        """Load system prompt"""
        prompt_path = "src/prompts/prompt.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = "You are a Hive game AI. Predict the next move and board state in JSON format."
    
    @torch.inference_mode()
    def _get_board_embedding(self, board: Board) -> Optional[torch.Tensor]:
        """Get board embedding from GNN"""
        try:
            data = self.gnn_model._data_from_board(board)
            if data is None:
                return None
            data = data.to(self.device)
            emb = self.gnn_model.network.return_embedding(data)
            return emb.detach().cpu()
        except Exception as e:
            logger.warning(f"Failed to get board embedding: {e}")
            return None
            
    @torch.inference_mode()
    def _get_move_embedding(self, board: Board, move: Move) -> Optional[torch.Tensor]:
        """Get move embedding (difference between before and after board states)"""
        try:
            board_before = self._get_board_embedding(board)
            if board_before is None:
                return None
                
            board.safe_play(move)
            board_after = self._get_board_embedding(board)
            board.undo()
            
            if board_after is None:
                return None
                
            return board_after - board_before
        except Exception as e:
            logger.warning(f"Failed to get move embedding: {e}")
            return None
            
    def _embedding_to_cluster_id(self, embedding: torch.Tensor, centroids: torch.Tensor) -> int:
        """Convert embedding to cluster ID using cosine similarity"""
        if embedding is None or centroids is None:
            return 0
            
        # Normalize embeddings
        emb_norm = F.normalize(embedding.flatten(), p=2, dim=0)
        cent_norm = F.normalize(centroids, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.matmul(cent_norm, emb_norm)
        return similarities.argmax().item()
        
    def _board_to_cluster_token(self, board: Board) -> str:
        """Convert board state to cluster token"""
        embedding = self._get_board_embedding(board)
        if embedding is None or self.board_centroids is None:
            return "<BCL_0>"
            
        cluster_id = self._embedding_to_cluster_id(embedding, self.board_centroids)
        return self.board_id_to_token.get(cluster_id, "<BCL_0>")
        
    def _move_to_cluster_token(self, board: Board, move: Move) -> str:
        """Convert move to cluster token"""
        embedding = self._get_move_embedding(board, move)
        if embedding is None or self.move_centroids is None:
            return "<MCL_0>"
            
        cluster_id = self._embedding_to_cluster_id(embedding, self.move_centroids)
        return self.move_id_to_token.get(cluster_id, "<MCL_0>")
        
    def _cluster_token_to_move(self, board: Board, move_token: str) -> Optional[Move]:
        """Convert cluster token back to actual move by finding best matching legal move"""
        if move_token not in self.move_token_to_id:
            return None
            
        cluster_id = self.move_token_to_id[move_token]
        if cluster_id >= len(self.move_centroids):
            return None
            
        target_embedding = self.move_centroids[cluster_id]
        valid_moves = list(board.get_valid_moves())
        
        if not valid_moves:
            return None
            
        best_move = None
        best_similarity = -1.0
        
        for move in valid_moves:
            move_embedding = self._get_move_embedding(board, move)
            if move_embedding is not None:
                similarity = F.cosine_similarity(
                    move_embedding.flatten(), 
                    target_embedding.flatten(), 
                    dim=0
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_move = move
                    
        return best_move
        
    def _build_context_prompt(self, move_history: List[Tuple[str, str]], board: Board) -> str:
        """Build context prompt with game history"""
        context_parts = []
        
        # Add move history
        for i, (board_token, move_token) in enumerate(move_history):
            player = "Player 1" if i % 2 == 0 else "Player 2"
            context_parts.append(f"{player}: {board_token} {move_token}")
            
        # Add current board state
        context_parts.append(" Current board state: ")
        current_player = "Player 1" if len(move_history) % 2 == 0 else "Player 2"
        current_board_token = self._board_to_cluster_token(board)
        context_parts.append(f"{current_player}: {current_board_token}")
        
        # Add legal moves
        valid_moves = list(board.get_valid_moves())
        context_parts.append(" Choose ONE move among the following LEGAL moves: ")
        
        for move in valid_moves:
            move_token = self._move_to_cluster_token(board, move)
            context_parts.append(f"- {move_token}")
            
        return "  ".join(context_parts)
        
    def detect_winning_move(self, board: Board) -> Optional[Move]:
        """
        Check if there's a winning move available in the current position.
        Returns the winning move if found, None otherwise.
        """
        valid_moves = list(board.get_valid_moves())
        current_player = board.current_player_color
        
        for move in valid_moves:
            # Try the move
            board.safe_play(move)
            
            # Check if this move wins the game
            is_winning = False
            if current_player == PlayerColor.WHITE and board.state == GameState.WHITE_WINS:
                is_winning = True
            elif current_player == PlayerColor.BLACK and board.state == GameState.BLACK_WINS:
                is_winning = True
                
            # Undo the move
            board.undo()
            
            if is_winning:
                logger.info(f"Winning move detected: {board.stringify_move(move)}")
                return move
                
        return None

    def predict_move(self, board: Board, move_history: List[Tuple[str, str]], 
                    check_winning_moves: bool = True) -> Tuple[Optional[Move], Dict]:
        """
        Predict next move given board state and history.
        
        Args:
            board: Current board state
            move_history: List of (board_token, move_token) tuples
            check_winning_moves: If True, check for immediate winning moves first
        
        Returns:
            Tuple of (predicted move, prediction data dictionary)
        """
        
        prediction_data = {
            "checked_for_winning_move": check_winning_moves,
            "winning_move_found": False
        }
        
        # First, check if there's a winning move available
        if check_winning_moves:
            winning_move = self.detect_winning_move(board)
            if winning_move is not None:
                # If winning move found, return it immediately
                move_token = self._move_to_cluster_token(board, winning_move)
                prediction_data.update({
                    "winning_move_found": True,
                    "move_token": move_token,
                    "board_token": self._board_to_cluster_token(board),
                    "predicted_move": winning_move,
                    "move_string": board.stringify_move(winning_move),
                    "method": "winning_move_detection"
                })
                logger.info(f"Playing winning move: {board.stringify_move(winning_move)}")
                return winning_move, prediction_data
        
        # No winning move found, proceed with LLM prediction
        # Build prompt
        context = self._build_context_prompt(move_history, board)
        
        # Create chat messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": context})
        
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            # Fallback to simple concatenation
            prompt = f"System: {self.system_prompt}\nUser: {context}\nAssistant: "
            
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            
        # Decode response
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        logger.info(f"Raw model response: {response}")
        
        # Update prediction data
        prediction_data.update({
            "raw_response": response,
            "context": context,
            "prompt": prompt,
            "method": "llm_prediction"
        })
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                move_token = parsed.get("move")
                board_token = parsed.get("board")
                
                prediction_data.update({
                    "parsed_json": parsed,
                    "move_token": move_token,
                    "board_token": board_token
                })
                
                # Convert move token to actual move
                if move_token and move_token != "null":
                    move = self._cluster_token_to_move(board, move_token)
                    
                    # Double-check: if this move is actually a winning move, note it
                    if move and check_winning_moves:
                        board.safe_play(move)
                        current_player = board.current_player_color
                        if (current_player == PlayerColor.WHITE and board.state == GameState.WHITE_WINS) or \
                        (current_player == PlayerColor.BLACK and board.state == GameState.BLACK_WINS):
                            prediction_data["llm_found_winning_move"] = True
                            logger.info("LLM independently found the winning move!")
                        board.undo()
                    
                    prediction_data["predicted_move"] = move
                    return move, prediction_data
                    
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            prediction_data["parse_error"] = str(e)
            
        # Fallback: return random legal move
        valid_moves = list(board.get_valid_moves())
        if valid_moves:
            import random
            
            # Even in fallback, prefer winning move if available
            if check_winning_moves:
                winning_move = self.detect_winning_move(board)
                if winning_move:
                    prediction_data.update({
                        "fallback_move": winning_move,
                        "fallback_was_winning": True,
                        "method": "fallback_winning"
                    })
                    return winning_move, prediction_data
            
            fallback_move = random.choice(valid_moves)
            prediction_data.update({
                "fallback_move": fallback_move,
                "method": "fallback_random"
            })
            return fallback_move, prediction_data
            
        return None, prediction_data


def play_game(llm_player: HiveLLMInference, random_player: Random, 
              llm_color: PlayerColor = PlayerColor.WHITE, max_moves: int = 200,
              enable_winning_detection: bool = True) -> Dict:
    """
    Play a full game between LLM and random player.
    
    Args:
        llm_player: The LLM player instance
        random_player: The random player instance
        llm_color: Color for the LLM player
        max_moves: Maximum number of moves before declaring a draw
        enable_winning_detection: If True, LLM will check for winning moves before prediction
    
    Returns:
        Dictionary containing game results and statistics
    """
    
    board = Board()
    move_history = []  # List of (board_token, move_token) tuples
    game_log = []
    winning_moves_detected = 0
    
    logger.info(f"Starting game: LLM as {llm_color}, Random as {PlayerColor.BLACK if llm_color == PlayerColor.WHITE else PlayerColor.WHITE}")
    logger.info(f"Winning move detection: {'ENABLED' if enable_winning_detection else 'DISABLED'}")
    
    move_count = 0
    while board.state in (GameState.NOT_STARTED, GameState.IN_PROGRESS) and move_count < max_moves:
        current_player = board.current_player_color
        
        if current_player == llm_color:
            # LLM's turn
            logger.info(f"Move {move_count + 1}: LLM ({current_player}) thinking...")
            
            # Pass the enable_winning_detection flag to predict_move
            move, prediction_data = llm_player.predict_move(
                board, 
                move_history,
                check_winning_moves=enable_winning_detection
            )
            
            if move is None:
                logger.error("LLM failed to predict a valid move!")
                break
            
            # Track if this was a winning move detection
            if prediction_data.get("winning_move_found", False):
                winning_moves_detected += 1
                logger.info(f"🎯 WINNING MOVE DETECTED AND PLAYED! Total detected: {winning_moves_detected}")
                
            move_str = board.stringify_move(move)
            logger.info(f"LLM plays: {move_str} (method: {prediction_data.get('method', 'unknown')})")
            
            # Get tokens for history
            board_token_before = llm_player._board_to_cluster_token(board)
            move_token = llm_player._move_to_cluster_token(board, move)
            
            # Play move
            board.safe_play(move)
            
            # Log turn data
            turn_data = {
                "move_number": move_count + 1,
                "player": "LLM",
                "color": str(current_player),
                "move": move_str,
                "board_token": board_token_before,
                "move_token": move_token,
                "board_state_after": str(board),
                "prediction_data": prediction_data,
                "was_winning_move": prediction_data.get("winning_move_found", False)
            }
            game_log.append(turn_data)
            move_history.append((board_token_before, move_token))
            
        else:
            # Random player's turn
            logger.info(f"Move {move_count + 1}: Random ({current_player}) thinking...")
            
            move_str = random_player.calculate_best_move(board, "depth", 1)
            logger.info(f"Random plays: {move_str}")
            
            # Get tokens for history
            board_token_before = llm_player._board_to_cluster_token(board)
            
            # Parse and play move
            if move_str.lower() != "pass":
                move = board._parse_move(move_str)
                move_token = llm_player._move_to_cluster_token(board, move)
                board.play(move_str)
            else:
                move_token = "<MCL_0>"  # Default for pass
                board.play(move_str)
                
            # Log turn data
            turn_data = {
                "move_number": move_count + 1,
                "player": "Random",
                "color": str(current_player),
                "move": move_str,
                "board_token": board_token_before,
                "move_token": move_token,
                "board_state_after": str(board)
            }
            game_log.append(turn_data)
            move_history.append((board_token_before, move_token))
            
        move_count += 1
        
    # Game finished
    final_state = board.state
    logger.info(f"Game finished after {move_count} moves. Result: {final_state}")
    logger.info(f"Winning moves detected by LLM: {winning_moves_detected}")
    
    return {
        "final_state": str(final_state),
        "total_moves": move_count,
        "move_history": move_history,
        "game_log": game_log,
        "winning_moves_detected": winning_moves_detected,
        "winner": "LLM" if (final_state == GameState.WHITE_WINS and llm_color == PlayerColor.WHITE) or 
                           (final_state == GameState.BLACK_WINS and llm_color == PlayerColor.BLACK) 
                  else "Random" if final_state in (GameState.WHITE_WINS, GameState.BLACK_WINS)
                  else "Draw"
    }


def main():
    parser = argparse.ArgumentParser(description="LLM vs Random Hive Game Inference")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--gnn-model", required=True, help="Path to GNN model (.pt file)")
    parser.add_argument("--num-games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--llm-color", choices=["white", "black"], default="white", 
                       help="Color for LLM player")
    parser.add_argument("--output-dir", default="inference_results", help="Directory to save results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum moves per game")
    parser.add_argument("--disable-winning-detection", action="store_true",
                       help="Disable automatic winning move detection")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    llm_color = PlayerColor.WHITE if args.llm_color == "white" else PlayerColor.BLACK
    enable_winning_detection = not args.disable_winning_detection
    
    # Initialize players
    logger.info("Initializing LLM player...")
    llm_player = HiveLLMInference(args.model_dir, args.gnn_model, args.device)
    random_player = Random()
    
    # Play games
    results = []
    wins = {"LLM": 0, "Random": 0, "Draw": 0}
    total_winning_moves_detected = 0
    games_with_winning_moves = 0
    
    for game_num in range(args.num_games):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Game {game_num + 1}/{args.num_games}")
        logger.info(f"{'='*50}")
        
        game_result = play_game(
            llm_player, 
            random_player, 
            llm_color, 
            args.max_moves,
            enable_winning_detection=enable_winning_detection
        )
        game_result["game_number"] = game_num + 1
        results.append(game_result)
        
        winner = game_result["winner"]
        wins[winner] += 1
        
        # Track winning move statistics
        winning_moves_in_game = game_result.get("winning_moves_detected", 0)
        if winning_moves_in_game > 0:
            games_with_winning_moves += 1
            total_winning_moves_detected += winning_moves_in_game
        
        logger.info(f"Game {game_num + 1} finished. Winner: {winner}")
        if winning_moves_in_game > 0:
            logger.info(f"  → {winning_moves_in_game} winning move(s) detected in this game")
        
        # Save individual game result
        game_file = os.path.join(args.output_dir, f"game_{game_num + 1}.json")
        with open(game_file, 'w') as f:
            json.dump(game_result, f, indent=2, default=str)
            
    # Calculate statistics
    avg_winning_moves_per_game = total_winning_moves_detected / args.num_games if args.num_games > 0 else 0
    pct_games_with_winning_moves = (games_with_winning_moves / args.num_games * 100) if args.num_games > 0 else 0
    
    # Save overall results
    summary = {
        "total_games": args.num_games,
        "llm_color": args.llm_color,
        "winning_detection_enabled": enable_winning_detection,
        "wins": wins,
        "win_rate": wins["LLM"] / args.num_games if args.num_games > 0 else 0,
        "winning_move_stats": {
            "total_detected": total_winning_moves_detected,
            "games_with_winning_moves": games_with_winning_moves,
            "avg_per_game": avg_winning_moves_per_game,
            "pct_games_with_winning_moves": pct_games_with_winning_moves
        },
        "results": results
    }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
        
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Total games: {args.num_games}")
    logger.info(f"LLM wins: {wins['LLM']} ({wins['LLM']/args.num_games*100:.1f}%)")
    logger.info(f"Random wins: {wins['Random']} ({wins['Random']/args.num_games*100:.1f}%)")
    logger.info(f"Draws: {wins['Draw']} ({wins['Draw']/args.num_games*100:.1f}%)")
    
    if enable_winning_detection:
        logger.info(f"\n{'='*50}")
        logger.info("WINNING MOVE DETECTION STATISTICS")
        logger.info(f"{'='*50}")
        logger.info(f"Total winning moves detected: {total_winning_moves_detected}")
        logger.info(f"Games with winning moves: {games_with_winning_moves}/{args.num_games} ({pct_games_with_winning_moves:.1f}%)")
        logger.info(f"Average winning moves per game: {avg_winning_moves_per_game:.2f}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()