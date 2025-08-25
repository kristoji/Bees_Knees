from engineer import Engine
from ai.oracleGNN import OracleGNN
from ai.oracleRND import OracleRND
import os
from ai.log_utils import reset_log, log_header, log_subheader
# Removed trainer import to avoid working directory change
import re
import torch
import argparse
import shutil
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import pickle
from glob import glob
from tqdm import tqdm
from engine.board import Board
from engine.game import Move
from engine.enums import GameState
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter


class GameParser:
    """Parse game board files to extract moves and game states"""
    
    @staticmethod
    def parse_game_file(file_path: str) -> Tuple[str, List[str], str]:
        """
        Parse a game board file and extract initial state, moves, and winner.
        
        Args:
            file_path: Path to the game board file
            
        Returns:
            Tuple of (initial_gamestring, list_of_moves, winner)
            winner is "Black", "White", or "Unknown"
        """
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # Split by semicolons
        parts = content.split(';')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid game format in {file_path}")
        
        # First three parts are: game_type, game_state, turn_info
        game_type = parts[0]
        game_state = parts[1] 
        turn_info = parts[2]
        moves = parts[3:] if len(parts) > 3 else []
        
        # Extract winner information from game_state
        winner = "Unknown"
        if "BlackWins" in game_state:
            winner = "Black"
        elif "WhiteWins" in game_state:
            winner = "White"
        
        # Create initial gamestring (before any moves)
        initial_gamestring = f"{game_type};NotStarted;White[1]"
        
        return initial_gamestring, moves, winner
    
    @staticmethod
    def create_board_from_gamestring(gamestring: str) -> Board:
        """Create a board instance from a gamestring"""
        return Board(gamestring)


class LLMDataset(Dataset):
    """
    Dataset for LLM training where each sample consists of:
    [gnn_embedding of the board, list of embeddings of legal moves, chosen_move_index]
    """
    
    def __init__(self, gnn_model_path: str, game_files_dir: str, device: str = "cuda", 
                 sampling_strategy: str = "distributed", max_moves_per_game: int = 15,
                 move_embedding_mode: str = "difference", winner_only: bool = False):
        """
        Initialize the dataset.
        
        Args:
            gnn_model_path: Path to the trained GNN model (.pt file)
            game_files_dir: Directory containing game_*_board.txt files
            device: Device to run the GNN on
            sampling_strategy: How to sample moves from games
                - "distributed": Sample moves evenly across the game (opening, middle, endgame)
                - "all": Process all moves (up to max_moves_per_game)
                - "random": Randomly sample moves from the game
            max_moves_per_game: Maximum number of moves to process per game
            move_embedding_mode: How to compute move embeddings
                - "difference": embedding_after - embedding_before (default, smaller size)
                - "concatenation": concat(embedding_before, embedding_after) (larger size, 2x)
            winner_only: If True, only extract samples from winning player moves
        """
        self.device = device
        self.game_files_dir = game_files_dir
        self.sampling_strategy = sampling_strategy
        self.max_moves_per_game = max_moves_per_game
        self.move_embedding_mode = move_embedding_mode
        self.winner_only = winner_only
        
        # Load the GNN model
        log_header("Loading GNN model")
        # Use the same configuration as the saved model (from train_gnn.py)
        kwargs_network = {
        # Architecture options
        'conv_type': 'GIN',  # 'GIN', 'GAT', 'GCN'
        'num_layers': 6,
        # GAT specific options
        'gat_heads': 4,
        'gat_concat': True,
        # Dropout options
        'conv_dropout': 0.1,
        'mlp_dropout': 0.1,
        'final_dropout': 0.2,
        # Normalization options
        'use_batch_norm': False,
        'use_layer_norm': True,
        # Residual connections
        'use_residual': False,
        # Pooling options
        'pooling': 'add',  # 'mean', 'max', 'add', 'concat'
        # MLP options
        'mlp_layers': 3,
        'final_mlp_layers': 3
        }
        self.gnn_model = OracleGNN(device=str(device), hidden_dim=256, **kwargs_network)  # Initialize the OracleGNN
        self.gnn_model.load(gnn_model_path)
        self.gnn_model.network.eval()

        # Find all game files
        self.game_files = glob(os.path.join(game_files_dir, "game_*_board.txt"))
        log_subheader(f"Found {len(self.game_files)} game files")
        
        # Process all games and create samples
        self.samples = []
        self._process_all_games()
        
        log_subheader(f"Created {len(self.samples)} training samples")
        
        # Print debugging information for first few samples
        self._debug_print_samples(num_samples=3)
    
    def _process_all_games(self):
        """Process all game files and create training samples"""
        parser = GameParser()
        
        total_moves_processed = 0
        total_valid_embeddings = 0
        total_valid_moves_found = 0
        total_chosen_moves_found = 0
        
        for game_file in tqdm(self.game_files, desc="Processing games"):
            try:
                initial_gamestring, moves, winner = parser.parse_game_file(game_file)
                board = parser.create_board_from_gamestring(initial_gamestring)
                
                log_subheader(f"Processing {os.path.basename(game_file)}: {len(moves)} moves")
                
                # Determine which moves to sample based on strategy
                moves_to_sample = self._get_moves_to_sample(len(moves))
                log_subheader(f"  Sampling strategy: {self.sampling_strategy}")
                log_subheader(f"  Selected {len(moves_to_sample)} moves from {len(moves)} total: {moves_to_sample}")
                
                # Play moves up to the first sampled move to set up board state
                current_move_idx = 0
                
                # Process each selected move
                for target_move_idx in moves_to_sample:
                    # Play moves to reach the target position (but not the target move itself)
                    while current_move_idx < target_move_idx:
                        move_str = moves[current_move_idx]
                        total_moves_processed += 1
                        
                        if move_str.strip() == "pass":
                            log_subheader(f"  Move {current_move_idx}: PASS - playing")
                        else:
                            log_subheader(f"  Move {current_move_idx}: '{move_str}' - playing to reach position")
                        
                        try:
                            board.play(move_str)
                        except Exception as e:
                            log_subheader(f"    Error playing move {move_str}: {e}")
                            # Skip this game if we can't reach the position
                            break
                        current_move_idx += 1
                    
                    # Check if we successfully reached the target position
                    if current_move_idx != target_move_idx:
                        log_subheader(f"  Failed to reach move {target_move_idx}, skipping remaining moves")
                        break
                    
                    # Now process the target move
                    move_str = moves[target_move_idx]
                    
                    # Check if we should skip this move based on winner_only filter
                    if self.winner_only:
                        # Determine current player (moves are 0-indexed, so even = white, odd = black)
                        current_player = "White" if target_move_idx % 2 == 0 else "Black"
                        if current_player != winner:
                            log_subheader(f"  Move {target_move_idx}: '{move_str}' - skipping (not winning player: {current_player} vs winner: {winner})")
                            # Still advance the current position
                            try:
                                board.play(move_str)
                                current_move_idx += 1
                            except Exception as e:
                                log_subheader(f"    Error playing move {move_str}: {e}")
                                break
                            continue
                    
                    if move_str.strip() == "pass":
                        log_subheader(f"  Move {target_move_idx}: PASS - skipping (no embedding generation)")
                        # Still advance the current position
                        try:
                            board.play(move_str)
                            current_move_idx += 1
                        except Exception as e:
                            log_subheader(f"    Error playing PASS: {e}")
                            break
                        continue
                        
                    # Skip if game is already finished
                    if board.state != GameState.IN_PROGRESS and board.state != GameState.NOT_STARTED:
                        log_subheader(f"  Move {target_move_idx}: Game finished - breaking")
                        break
                    
                    log_subheader(f"  Move {target_move_idx}: '{move_str}' - processing for dataset")
                    
                    # Get current board embedding
                    board_embedding = self._get_board_embedding(board)
                    if board_embedding is None:
                        log_subheader(f"    Board embedding: FAILED")
                        continue
                    else:
                        log_subheader(f"    Board embedding: OK (shape: {board_embedding.shape})")
                    
                    # Get valid moves and their embeddings
                    valid_moves = list(board.get_valid_moves())
                    if not valid_moves:
                        log_subheader(f"    Valid moves: NONE - skipping")
                        continue
                    else:
                        log_subheader(f"    Valid moves: {len(valid_moves)} found")
                        total_valid_moves_found += 1
                    
                    # Get embeddings and texts for all legal moves
                    move_result = self._get_move_embeddings_and_texts(board, valid_moves)
                    if move_result is None:
                        log_subheader(f"    Move embeddings: FAILED")
                        continue
                    else:
                        move_embeddings, move_texts, next_board_embeddings = move_result
                        log_subheader(f"    Move embeddings: OK (shape: {move_embeddings.shape})")
                        log_subheader(f"    Move texts: {len(move_texts)} descriptions")
                        log_subheader(f"    Next board embeddings: OK (shape: {next_board_embeddings.shape})")
                        total_valid_embeddings += 1
                    
                    # Find the chosen move index
                    chosen_move_idx = self._find_chosen_move_index(board, move_str, valid_moves)
                    if chosen_move_idx is None:
                        log_subheader(f"    Chosen move index: NOT FOUND for '{move_str}'")
                        
                        # Debug: show what valid moves look like
                        log_subheader(f"    Available moves:")
                        for i, vm in enumerate(valid_moves[:5]):  # Show first 5
                            vm_str = board.stringify_move(vm)
                            log_subheader(f"      {i}: {vm_str}")
                        if len(valid_moves) > 5:
                            log_subheader(f"      ... and {len(valid_moves)-5} more")
                        
                        # Still advance the position even if we can't process this move
                        try:
                            board.play(move_str)
                            current_move_idx += 1
                        except Exception as e:
                            log_subheader(f"    Error playing move {move_str}: {e}")
                            break
                        continue
                    else:
                        log_subheader(f"    Chosen move index: {chosen_move_idx}")
                        total_chosen_moves_found += 1
                    
                    if chosen_move_idx is not None and board_embedding is not None and move_embeddings is not None:
                        # Extract the chosen move embedding
                        chosen_move_embedding = move_embeddings[chosen_move_idx]
                        
                        # Extract the board state embedding after the chosen move
                        chosen_next_board_embedding = next_board_embeddings[chosen_move_idx]
                        
                        # Determine game phase for metadata
                        game_phase = "opening"
                        if target_move_idx > len(moves) * 2 // 3:
                            game_phase = "endgame"
                        elif target_move_idx > len(moves) // 3:
                            game_phase = "midgame"
                        
                        sample = {
                            'board_embedding': board_embedding,
                            'move_embeddings': move_embeddings,
                            'move_texts': move_texts,
                            'next_board_embeddings': next_board_embeddings,
                            'chosen_move_idx': chosen_move_idx,
                            'chosen_move_embedding': chosen_move_embedding,
                            'chosen_next_board_embedding': chosen_next_board_embedding,
                            'chosen_move_text': move_str,
                            'game_file': os.path.basename(game_file),
                            'move_number': target_move_idx,
                            'move_string': move_str,
                            'game_phase': game_phase,
                            'total_game_moves': len(moves)
                        }
                        self.samples.append(sample)
                        log_subheader(f"    SAMPLE CREATED ({game_phase})! Total samples: {len(self.samples)}")
                    
                    # Play the move to advance the board state
                    try:
                        board.play(move_str)
                        current_move_idx += 1
                        log_subheader(f"    Move played successfully")
                    except Exception as e:
                        log_subheader(f"    Error playing move {move_str}: {e}")
                        break
                        
            except Exception as e:
                log_subheader(f"Error processing {game_file}: {e}")
                continue
        
        # Print summary statistics
        log_header("=== PROCESSING SUMMARY ===")
        log_subheader(f"Total moves processed: {total_moves_processed}")
        log_subheader(f"Valid board embeddings: {total_valid_embeddings}")
        log_subheader(f"Valid move sets found: {total_valid_moves_found}")
        log_subheader(f"Chosen moves found: {total_chosen_moves_found}")
        log_subheader(f"Final samples created: {len(self.samples)}")
        log_header("=== END SUMMARY ===")
    
    def _debug_print_samples(self, num_samples: int = 3):
        """Print debugging information for first few samples"""
        if not self.samples:
            log_subheader("No samples to debug - dataset is empty")
            return
        
        log_header("=== DATASET DEBUGGING INFO ===")
        
        # Print overall statistics
        total_samples = len(self.samples)
        log_subheader(f"Total samples in dataset: {total_samples}")
        
        # Get dimensions info
        if total_samples > 0:
            first_sample = self.samples[0]
            if first_sample['board_embedding'] is not None:
                board_emb_dim = first_sample['board_embedding'].shape[-1]
                log_subheader(f"Board embedding dimension: {board_emb_dim}")
            
            if first_sample['move_embeddings'] is not None:
                move_emb_shape = first_sample['move_embeddings'].shape
                log_subheader(f"Move embeddings shape example: {move_emb_shape}")
        
        # Print detailed info for first few samples
        num_to_show = min(num_samples, total_samples)
        log_subheader(f"Showing details for first {num_to_show} samples:")
        
        for i in range(num_to_show):
            sample = self.samples[i]
            log_subheader(f"\n--- Sample {i+1} ---")
            log_subheader(f"Game file: {sample['game_file']}")
            log_subheader(f"Move number: {sample['move_number']}")
            log_subheader(f"Move string: '{sample['move_string']}'")
            log_subheader(f"Chosen move index: {sample['chosen_move_idx']}")
            
            # Board embedding info
            if sample['board_embedding'] is not None:
                board_emb = sample['board_embedding']
                log_subheader(f"Board embedding shape: {board_emb.shape}")
                log_subheader(f"Board embedding stats: min={board_emb.min():.4f}, max={board_emb.max():.4f}, mean={board_emb.mean():.4f}")
                log_subheader(f"Board embedding sample values: {board_emb.flatten()[:5].tolist()}")
            else:
                log_subheader("Board embedding: None")
            
            # Move embeddings info
            if sample['move_embeddings'] is not None:
                move_embs = sample['move_embeddings']
                log_subheader(f"Move embeddings shape: {move_embs.shape}")
                log_subheader(f"Number of legal moves: {move_embs.shape[0]}")
                log_subheader(f"Move embeddings stats: min={move_embs.min():.4f}, max={move_embs.max():.4f}, mean={move_embs.mean():.4f}")
                
                # Show stats for chosen move
                if 0 <= sample['chosen_move_idx'] < move_embs.shape[0]:
                    chosen_move_emb = move_embs[sample['chosen_move_idx']]
                    log_subheader(f"Chosen move embedding stats: min={chosen_move_emb.min():.4f}, max={chosen_move_emb.max():.4f}, mean={chosen_move_emb.mean():.4f}")
                    log_subheader(f"Chosen move embedding sample values: {chosen_move_emb.flatten()[:5].tolist()}")
                else:
                    log_subheader(f"WARNING: Chosen move index {sample['chosen_move_idx']} is out of range for {move_embs.shape[0]} moves")
            else:
                log_subheader("Move embeddings: None")
        
        # Print distribution of number of legal moves
        move_counts = [s['move_embeddings'].shape[0] for s in self.samples if s['move_embeddings'] is not None]
        if move_counts:
            log_subheader(f"\nLegal moves distribution:")
            log_subheader(f"  Min legal moves: {min(move_counts)}")
            log_subheader(f"  Max legal moves: {max(move_counts)}")
            log_subheader(f"  Average legal moves: {np.mean(move_counts):.2f}")
            log_subheader(f"  Median legal moves: {np.median(move_counts):.2f}")
        
        # Print distribution of chosen move indices
        chosen_indices = [s['chosen_move_idx'] for s in self.samples if s['chosen_move_idx'] is not None]
        if chosen_indices:
            log_subheader(f"\nChosen move index distribution:")
            log_subheader(f"  Min index: {min(chosen_indices)}")
            log_subheader(f"  Max index: {max(chosen_indices)}")
            log_subheader(f"  Average index: {np.mean(chosen_indices):.2f}")
        
        log_header("=== END DEBUGGING INFO ===")
    
    def _get_moves_to_sample(self, total_moves: int) -> List[int]:
        """
        Determine which move indices to sample based on the sampling strategy
        
        Args:
            total_moves: Total number of moves in the game
            
        Returns:
            List of move indices to process (1-indexed, skipping move 0)
        """
        if total_moves <= 0:
            return []
        
        # Available moves (include move 0 which is the starting move)
        available_moves = list(range(0, total_moves))
        
        if self.sampling_strategy == "all":
            # Take all moves up to max_moves_per_game
            return available_moves[:self.max_moves_per_game]
        
        elif self.sampling_strategy == "distributed":
            # Sample moves evenly distributed across the game
            if len(available_moves) <= self.max_moves_per_game:
                return available_moves
            
            # Divide game into phases and sample from each
            game_length = len(available_moves)
            
            if game_length <= 6:
                # Short game - take all moves
                return available_moves
            
            # Define game phases
            opening_end = min(6, game_length // 3)
            midgame_end = min(game_length - 3, 2 * game_length // 3)
            
            opening_moves = available_moves[:opening_end]
            midgame_moves = available_moves[opening_end:midgame_end]
            endgame_moves = available_moves[midgame_end:]
            
            # Sample from each phase
            moves_per_phase = self.max_moves_per_game // 3
            remainder = self.max_moves_per_game % 3
            
            selected_moves = []
            
            # Opening moves (sample evenly)
            if opening_moves:
                n_opening = moves_per_phase + (1 if remainder > 0 else 0)
                remainder = max(0, remainder - 1)
                step = max(1, len(opening_moves) // n_opening)
                selected_moves.extend(opening_moves[::step][:n_opening])
            
            # Midgame moves (sample evenly)
            if midgame_moves:
                n_midgame = moves_per_phase + (1 if remainder > 0 else 0)
                remainder = max(0, remainder - 1)
                step = max(1, len(midgame_moves) // n_midgame)
                selected_moves.extend(midgame_moves[::step][:n_midgame])
            
            # Endgame moves (sample evenly)
            if endgame_moves:
                n_endgame = moves_per_phase + (1 if remainder > 0 else 0)
                step = max(1, len(endgame_moves) // n_endgame)
                selected_moves.extend(endgame_moves[::step][:n_endgame])
            
            return sorted(selected_moves)
        
        elif self.sampling_strategy == "random":
            # Randomly sample moves
            import random
            if len(available_moves) <= self.max_moves_per_game:
                return available_moves
            return sorted(random.sample(available_moves, self.max_moves_per_game))
        
        elif self.sampling_strategy == "unlimited":
            # Take ALL moves regardless of max_moves_per_game limit
            return available_moves
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def _get_board_embedding(self, board: Board) -> Optional[torch.Tensor]:
        """Get GNN embedding for the current board state"""
        try:
            with torch.no_grad():
                # Convert board to graph data
                data = self.gnn_model._data_from_board(board)
                if data is None:
                    return None
                
                # Move to device and add batch dimension
                data = data.to(self.device)
                
                # Get embedding from GNN
                embedding = self.gnn_model.network.return_embedding(data)
                return embedding.cpu()
                
        except Exception as e:
            return None
    
    def _get_move_embeddings_and_texts(self, board: Board, valid_moves: List[Move]) -> Optional[Tuple[torch.Tensor, List[str], torch.Tensor]]:
        """
        Get embeddings and textual descriptions for all legal moves.
        Move embedding calculation depends on self.move_embedding_mode:
        - "difference": embedding_after_move - embedding_before_move (default)
        - "concatenation": concat(embedding_before_move, embedding_after_move)
        
        Returns:
            Tuple of (move_embeddings_tensor, move_text_descriptions, next_board_embeddings_tensor)
        """
        try:
            board_embedding_before = self._get_board_embedding(board)
            if board_embedding_before is None:
                return None
            
            move_embeddings = []
            move_texts = []
            next_board_embeddings = []
            
            for move in valid_moves:
                try:
                    # Get textual description of the move
                    move_text = board.stringify_move(move)
                    move_texts.append(move_text)
                    
                    # Play the move
                    board.safe_play(move)  # This expects a Move object, which is correct here
                    
                    # Get embedding after the move
                    board_embedding_after = self._get_board_embedding(board)
                    
                    # Undo the move
                    board.undo()
                    
                    if board_embedding_after is not None:
                        # Calculate move embedding based on mode
                        if self.move_embedding_mode == "concatenation":
                            # Concatenate before and after embeddings
                            move_embedding = torch.cat([board_embedding_before, board_embedding_after], dim=-1)
                        else:  # default "difference" mode
                            # Calculate difference embedding
                            move_embedding = board_embedding_after - board_embedding_before
                        
                        move_embeddings.append(move_embedding)
                        # Store the full next board embedding
                        next_board_embeddings.append(board_embedding_after)
                    else:
                        # If we can't get embedding after move, use zero vector
                        if self.move_embedding_mode == "concatenation":
                            # Double size zero vector for concatenation mode
                            zero_embedding = torch.cat([torch.zeros_like(board_embedding_before), 
                                                       torch.zeros_like(board_embedding_before)], dim=-1)
                        else:
                            zero_embedding = torch.zeros_like(board_embedding_before)
                        
                        move_embeddings.append(zero_embedding)
                        next_board_embeddings.append(torch.zeros_like(board_embedding_before))
                        
                except Exception as e:
                    # If there's an error with this specific move, skip it or use zero vector
                    if self.move_embedding_mode == "concatenation":
                        # Double size zero vector for concatenation mode
                        zero_embedding = torch.cat([torch.zeros_like(board_embedding_before), 
                                                   torch.zeros_like(board_embedding_before)], dim=-1)
                    else:
                        zero_embedding = torch.zeros_like(board_embedding_before)
                    
                    move_embeddings.append(zero_embedding)
                    next_board_embeddings.append(torch.zeros_like(board_embedding_before))
                    # Still add the move text even if embedding failed
                    if len(move_texts) < len(move_embeddings):
                        move_texts.append(f"ERROR_MOVE_{len(move_texts)}")
                    log_subheader(f"Error processing move {move}: {e}")
            
            if move_embeddings:
                return torch.stack(move_embeddings), move_texts, torch.stack(next_board_embeddings)
            else:
                return None
                
        except Exception as e:
            return None
    
    def _get_move_embeddings(self, board: Board, valid_moves: List[Move]) -> Optional[torch.Tensor]:
        """
        Get embeddings for all legal moves.
        Each move embedding is: embedding_after_move - embedding_before_move
        """
        result = self._get_move_embeddings_and_texts(board, valid_moves)
        if result is not None:
            return result[0]  # Return only embeddings for backward compatibility
        return None
    
    def _find_chosen_move_index(self, board: Board, move_str: str, valid_moves: List[Move]) -> Optional[int]:
        """Find the index of the chosen move in the list of valid moves"""
        try:
            # Parse the move string to get a Move object
            chosen_move = board._parse_move(move_str)
            if chosen_move is None:
                return None
            
            # Find the index in valid moves by comparing move properties
            for idx, valid_move in enumerate(valid_moves):
                # Compare the string representations since Move objects might not be exactly equal
                chosen_move_str = board.stringify_move(chosen_move)
                valid_move_str = board.stringify_move(valid_move)
                
                if chosen_move_str == valid_move_str:
                    return idx
                
                # Also try direct comparison
                if (chosen_move.bug == valid_move.bug and 
                    chosen_move.origin == valid_move.origin and 
                    chosen_move.destination == valid_move.destination):
                    return idx
            
            return None
            
        except Exception as e:
            log_subheader(f"Error finding chosen move '{move_str}': {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'board_embedding': sample['board_embedding'],
            'move_embeddings': sample['move_embeddings'],
            'move_texts': sample['move_texts'],
            'next_board_embeddings': sample['next_board_embeddings'],
            'chosen_move_idx': sample['chosen_move_idx'],
            'chosen_move_embedding': sample['chosen_move_embedding'],
            'chosen_next_board_embedding': sample['chosen_next_board_embedding'],
            'chosen_move_text': sample['chosen_move_text']
        }
    
    def save_to_cache(self, cache_path: str):
        """Save processed dataset to cache file"""
        log_header(f"Saving dataset to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.samples, f)
        log_subheader("Cache saved successfully")
    
    @classmethod
    def load_from_cache(cls, cache_path: str, gnn_model_path: str, device: str = "cuda", 
                       move_embedding_mode: str = "difference"):
        """Load dataset from cache file"""
        log_header(f"Loading dataset from cache: {cache_path}")
        
        # Create empty instance
        instance = cls.__new__(cls)
        instance.device = device
        instance.move_embedding_mode = move_embedding_mode
        
        # Load GNN model
        kwargs_network = {
            # Architecture options
            'conv_type': 'GIN',  # 'GIN', 'GAT', 'GCN'
            'num_layers': 2,
            # GAT specific options
            'gat_heads': 8,
            'gat_concat': True,
            # Dropout options
            'conv_dropout': 0.1,
            'mlp_dropout': 0.1,
            'final_dropout': 0.2,
            # Normalization options
            'use_batch_norm': False,
            'use_layer_norm': True,
            # Residual connections
            'use_residual': False,
            # Pooling options
            'pooling': 'add',  # 'mean', 'max', 'add', 'concat'
            # MLP options
            'mlp_layers': 2,
            'final_mlp_layers': 2
        }
        instance.gnn_model = OracleGNN(device=device, hidden_dim=24, **kwargs_network)
        instance.gnn_model.load(gnn_model_path)
        instance.gnn_model.network.eval()
        
        # Load samples from cache
        with open(cache_path, 'rb') as f:
            instance.samples = pickle.load(f)
        
        log_subheader(f"Loaded {len(instance.samples)} samples from cache")
        
        # Print debugging information for cached samples
        instance._debug_print_samples(num_samples=3)
        
        return instance
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length move sequences
        """
        # Extract components
        board_embeddings = [item['board_embedding'] for item in batch]
        move_embeddings = [item['move_embeddings'] for item in batch]
        move_texts_list = [item['move_texts'] for item in batch]
        next_board_embeddings_list = [item['next_board_embeddings'] for item in batch]
        chosen_move_indices = [item['chosen_move_idx'] for item in batch]
        chosen_move_embeddings = [item['chosen_move_embedding'] for item in batch]
        chosen_next_board_embeddings = [item['chosen_next_board_embedding'] for item in batch]
        chosen_move_texts = [item['chosen_move_text'] for item in batch]
        
        # Stack board embeddings (all same size)
        board_embeddings = torch.stack(board_embeddings, dim=0)
        
        # Stack chosen move embeddings (all same size)
        chosen_move_embeddings = torch.stack(chosen_move_embeddings, dim=0)
        
        # Stack chosen next board embeddings (all same size)
        chosen_next_board_embeddings = torch.stack(chosen_next_board_embeddings, dim=0)
        
        # For move embeddings and next board embeddings, we need to pad to the same length
        max_moves = max(me.size(0) for me in move_embeddings)
        padded_move_embeddings = []
        padded_next_board_embeddings = []
        padded_move_texts = []
        move_masks = []
        
        for i, (me, nbe, texts) in enumerate(zip(move_embeddings, next_board_embeddings_list, move_texts_list)):
            num_moves, embedding_size, move_hidden_dim = me.shape
            _, _, board_hidden_dim = nbe.shape
            
            # Create padded tensor for move embeddings
            padded_me = torch.zeros(max_moves, embedding_size, move_hidden_dim)
            padded_me[:num_moves] = me
            padded_move_embeddings.append(padded_me)
            
            # Create padded tensor for next board embeddings
            padded_nbe = torch.zeros(max_moves, embedding_size, board_hidden_dim)
            padded_nbe[:num_moves] = nbe
            padded_next_board_embeddings.append(padded_nbe)
            
            # Create padded list for texts
            padded_text_list = texts + ["<PAD>"] * (max_moves - len(texts))
            padded_move_texts.append(padded_text_list)
            
            # Create mask (1 for real moves, 0 for padding)
            mask = torch.zeros(max_moves, dtype=torch.bool)
            mask[:num_moves] = True
            move_masks.append(mask)
        
        move_embeddings = torch.stack(padded_move_embeddings, dim=0)
        next_board_embeddings = torch.stack(padded_next_board_embeddings, dim=0)
        move_masks = torch.stack(move_masks, dim=0)
        chosen_move_indices = torch.tensor(chosen_move_indices, dtype=torch.long)
        
        return {
            'board_embedding': board_embeddings,
            'move_embeddings': move_embeddings,
            'move_texts': padded_move_texts,
            'next_board_embeddings': next_board_embeddings,
            'chosen_move_idx': chosen_move_indices,
            'chosen_move_embedding': chosen_move_embeddings,
            'chosen_next_board_embedding': chosen_next_board_embeddings,
            'chosen_move_text': chosen_move_texts,
            'move_masks': move_masks  # Add masks to indicate valid moves
        }

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """Create a DataLoader for the dataset"""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, 
                         collate_fn=self.collate_fn, **kwargs)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.samples:
            return {}
        
        board_emb_dims = [s['board_embedding'].shape[-1] for s in self.samples if s['board_embedding'] is not None]
        move_counts = [s['move_embeddings'].shape[0] for s in self.samples if s['move_embeddings'] is not None]
        
        stats = {
            'total_samples': len(self.samples),
            'board_embedding_dim': board_emb_dims[0] if board_emb_dims else None,
            'avg_legal_moves': np.mean(move_counts) if move_counts else 0,
            'max_legal_moves': max(move_counts) if move_counts else 0,
            'min_legal_moves': min(move_counts) if move_counts else 0
        }
        
        return stats
    
    def save_to_csv(self, csv_path: str):
        """Save dataset statistics to CSV for analysis"""
        data = []
        for i, sample in enumerate(self.samples):
            row = {
                'sample_idx': i,
                'game_file': sample['game_file'],
                'move_number': sample['move_number'],
                'move_string': sample['move_string'],
                'chosen_move_idx': sample['chosen_move_idx'],
                'num_legal_moves': sample['move_embeddings'].shape[0] if sample['move_embeddings'] is not None else 0,
                'board_emb_dim': sample['board_embedding'].shape[-1] if sample['board_embedding'] is not None else 0,
                'game_phase': sample.get('game_phase', 'unknown'),
                'total_game_moves': sample.get('total_game_moves', 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        log_subheader(f"Saved dataset statistics to {csv_path}")
        
        # Print game phase distribution
        if 'game_phase' in df.columns:
            phase_counts = df['game_phase'].value_counts()
            log_subheader(f"Game phase distribution: {phase_counts.to_dict()}")
    
    def save_embeddings(self, output_dir: str):
        """
        Save the actual embeddings data to disk
        
        Args:
            output_dir: Directory to save embeddings
        """
        import os
        import pickle
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data structure
        embeddings_data = {
            'board_embeddings': [],
            'move_embeddings': [],
            'move_texts': [],
            'next_board_embeddings': [],
            'chosen_move_indices': [],
            'chosen_move_embeddings': [],
            'chosen_next_board_embeddings': [],
            'chosen_move_texts': [],
            'metadata': []
        }
        
        log_header("Saving embeddings to disk")
        
        for i, sample in enumerate(self.samples):
            # Board embedding (GNN embedding of current board state)
            board_emb = sample['board_embedding']
            if board_emb is not None:
                embeddings_data['board_embeddings'].append(board_emb.numpy())
            else:
                embeddings_data['board_embeddings'].append(None)
            
            # Move embeddings (embedding_after_move - embedding_before_move for each legal move)
            move_embs = sample['move_embeddings']
            if move_embs is not None:
                embeddings_data['move_embeddings'].append(move_embs.numpy())
            else:
                embeddings_data['move_embeddings'].append(None)
            
            # Move texts (textual descriptions of legal moves)
            move_texts = sample['move_texts']
            embeddings_data['move_texts'].append(move_texts)
            
            # Next board embeddings (GNN embeddings after each legal move)
            next_board_embs = sample['next_board_embeddings']
            if next_board_embs is not None:
                embeddings_data['next_board_embeddings'].append(next_board_embs.numpy())
            else:
                embeddings_data['next_board_embeddings'].append(None)
            
            # Chosen move index (ground truth)
            embeddings_data['chosen_move_indices'].append(sample['chosen_move_idx'])
            
            # Chosen move embedding
            chosen_move_emb = sample['chosen_move_embedding']
            if chosen_move_emb is not None:
                embeddings_data['chosen_move_embeddings'].append(chosen_move_emb.numpy())
            else:
                embeddings_data['chosen_move_embeddings'].append(None)
            
            # Chosen next board embedding (board state after chosen move)
            chosen_next_board_emb = sample['chosen_next_board_embedding']
            if chosen_next_board_emb is not None:
                embeddings_data['chosen_next_board_embeddings'].append(chosen_next_board_emb.numpy())
            else:
                embeddings_data['chosen_next_board_embeddings'].append(None)
            
            # Chosen move text (ground truth)
            embeddings_data['chosen_move_texts'].append(sample['chosen_move_text'])
            
            # Metadata
            metadata = {
                'game_file': sample['game_file'],
                'move_number': sample['move_number'],
                'move_string': sample['move_string'],
                'num_legal_moves': move_embs.shape[0] if move_embs is not None else 0,
                'game_phase': sample.get('game_phase', 'unknown'),
                'total_game_moves': sample.get('total_game_moves', 0)
            }
            embeddings_data['metadata'].append(metadata)
        
        # Save as pickle file
        embeddings_path = os.path.join(output_dir, 'llm_training_embeddings.pkl')
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        # Save as individual numpy files for easier loading
        board_embs_path = os.path.join(output_dir, 'board_embeddings.npy')
        move_embs_path = os.path.join(output_dir, 'move_embeddings.pkl')  # Variable length, use pickle
        move_texts_path = os.path.join(output_dir, 'move_texts.pkl')  # Variable length, use pickle
        next_board_embs_path = os.path.join(output_dir, 'next_board_embeddings.pkl')  # Variable length, use pickle
        chosen_indices_path = os.path.join(output_dir, 'chosen_move_indices.npy')
        chosen_embs_path = os.path.join(output_dir, 'chosen_move_embeddings.npy')
        chosen_next_board_embs_path = os.path.join(output_dir, 'chosen_next_board_embeddings.npy')
        chosen_texts_path = os.path.join(output_dir, 'chosen_move_texts.pkl')
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        
        # Save board embeddings (all same size, can use numpy)
        valid_board_embs = [emb for emb in embeddings_data['board_embeddings'] if emb is not None]
        if valid_board_embs:
            np.save(board_embs_path, np.stack(valid_board_embs))
            log_subheader(f"Saved {len(valid_board_embs)} board embeddings to {board_embs_path}")
        
        # Save move embeddings (variable length, use pickle)
        with open(move_embs_path, 'wb') as f:
            pickle.dump(embeddings_data['move_embeddings'], f)
        log_subheader(f"Saved move embeddings to {move_embs_path}")
        
        # Save move texts (variable length, use pickle)
        with open(move_texts_path, 'wb') as f:
            pickle.dump(embeddings_data['move_texts'], f)
        log_subheader(f"Saved move texts to {move_texts_path}")
        
        # Save next board embeddings (variable length, use pickle)
        with open(next_board_embs_path, 'wb') as f:
            pickle.dump(embeddings_data['next_board_embeddings'], f)
        log_subheader(f"Saved next board embeddings to {next_board_embs_path}")
        
        # Save chosen move indices
        np.save(chosen_indices_path, np.array(embeddings_data['chosen_move_indices']))
        log_subheader(f"Saved chosen move indices to {chosen_indices_path}")
        
        # Save chosen move embeddings (all same size, can use numpy)
        valid_chosen_embs = [emb for emb in embeddings_data['chosen_move_embeddings'] if emb is not None]
        if valid_chosen_embs:
            np.save(chosen_embs_path, np.stack(valid_chosen_embs))
            log_subheader(f"Saved {len(valid_chosen_embs)} chosen move embeddings to {chosen_embs_path}")
        
        # Save chosen next board embeddings (all same size, can use numpy)
        valid_chosen_next_board_embs = [emb for emb in embeddings_data['chosen_next_board_embeddings'] if emb is not None]
        if valid_chosen_next_board_embs:
            np.save(chosen_next_board_embs_path, np.stack(valid_chosen_next_board_embs))
            log_subheader(f"Saved {len(valid_chosen_next_board_embs)} chosen next board embeddings to {chosen_next_board_embs_path}")
        
        # Save chosen move texts
        with open(chosen_texts_path, 'wb') as f:
            pickle.dump(embeddings_data['chosen_move_texts'], f)
        log_subheader(f"Saved chosen move texts to {chosen_texts_path}")
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(embeddings_data['metadata'], f)
        log_subheader(f"Saved metadata to {metadata_path}")
        
        # Save data description
        description_path = os.path.join(output_dir, 'README.md')
        with open(description_path, 'w') as f:
            f.write("# LLM Training Dataset\n\n")
            f.write("This dataset contains embeddings for training an LLM on Hive game moves.\n\n")
            f.write("## Data Structure\n\n")
            f.write("Each sample consists of:\n")
            f.write("1. **Board embedding**: GNN embedding of the current board state (shape: [1, 24])\n")
            f.write(f"2. **Move embeddings**: List of embeddings for legal moves (shape: [N, 1, {24 if self.move_embedding_mode == 'difference' else 48}] where N = number of legal moves)\n")
            if self.move_embedding_mode == "concatenation":
                f.write("   - Each move embedding = concat(embedding_before_move, embedding_after_move) [CONCATENATION MODE]\n")
            else:
                f.write("   - Each move embedding = embedding_after_move - embedding_before_move [DIFFERENCE MODE]\n")
            if hasattr(self, 'winner_only') and self.winner_only:
                f.write("   - **WINNER ONLY MODE**: Only samples from winning player moves are included\n")
            f.write("   - **Move texts**: Corresponding textual descriptions of each legal move\n")
            f.write("3. **Next board embeddings**: List of GNN embeddings after each legal move (shape: [N, 1, 24])\n")
            f.write("   - Pre-computed board states after each legal move to avoid recalculation during training\n")
            f.write("4. **Chosen move embedding**: Embedding of the move that was actually played (shape: [1, 24])\n")
            f.write("5. **Chosen move text**: Textual representation of the chosen move (ground truth)\n\n")
            f.write("## Files\n\n")
            f.write("- `llm_training_embeddings.pkl`: Complete dataset in pickle format\n")
            f.write("- `board_embeddings.npy`: Board embeddings only (numpy array)\n")
            f.write("- `move_embeddings.pkl`: Move embeddings (variable length, pickle format)\n")
            f.write("- `move_texts.pkl`: Textual descriptions of legal moves (variable length, pickle format)\n")
            f.write("- `next_board_embeddings.pkl`: GNN embeddings after each legal move (variable length, pickle format)\n")
            f.write("- `chosen_move_indices.npy`: Ground truth move indices (numpy array)\n")
            f.write("- `chosen_move_embeddings.npy`: Chosen move embeddings (numpy array)\n")
            f.write("- `chosen_move_texts.pkl`: Chosen move text strings (pickle format)\n")
            f.write("- `metadata.pkl`: Sample metadata (game file, move number, etc.)\n")
            f.write("- `dataset_stats.csv`: Dataset statistics\n\n")
            f.write(f"## Dataset Statistics\n\n")
            f.write(f"- Total samples: {len(self.samples)}\n")
            f.write(f"- Board embedding dimension: {valid_board_embs[0].shape[-1] if valid_board_embs else 'N/A'}\n")
            move_counts = [emb.shape[0] for emb in embeddings_data['move_embeddings'] if emb is not None]
            if move_counts:
                f.write(f"- Legal moves range: {min(move_counts)}-{max(move_counts)} (avg: {np.mean(move_counts):.1f})\n")
        
        log_subheader(f"Saved data description to {description_path}")
        log_header(f"All embeddings saved to {output_dir}")
        
        return embeddings_path
    
    def debug_sample_content(self, sample_idx: int = 0):
        """Debug detailed content of a specific sample"""
        if sample_idx >= len(self.samples):
            log_subheader(f"Sample index {sample_idx} out of range (max: {len(self.samples)-1})")
            return
        
        sample = self.samples[sample_idx]
        log_header(f"=== DETAILED SAMPLE {sample_idx} DEBUG ===")
        
        log_subheader(f"Sample metadata:")
        log_subheader(f"  Game file: {sample['game_file']}")
        log_subheader(f"  Move number: {sample['move_number']}")
        log_subheader(f"  Move string: '{sample['move_string']}'")
        log_subheader(f"  Chosen move index: {sample['chosen_move_idx']}")
        
        # Board embedding detailed analysis
        if sample['board_embedding'] is not None:
            board_emb = sample['board_embedding']
            log_subheader(f"\nBoard embedding analysis:")
            log_subheader(f"  Shape: {board_emb.shape}")
            log_subheader(f"  Data type: {board_emb.dtype}")
            log_subheader(f"  Device: {board_emb.device}")
            log_subheader(f"  Min: {board_emb.min():.6f}")
            log_subheader(f"  Max: {board_emb.max():.6f}")
            log_subheader(f"  Mean: {board_emb.mean():.6f}")
            log_subheader(f"  Std: {board_emb.std():.6f}")
            log_subheader(f"  First 10 values: {board_emb.flatten()[:10].tolist()}")
            log_subheader(f"  Last 10 values: {board_emb.flatten()[-10:].tolist()}")
            
            # Check for problematic values
            nan_count = torch.isnan(board_emb).sum().item()
            inf_count = torch.isinf(board_emb).sum().item()
            zero_count = (board_emb == 0).sum().item()
            log_subheader(f"  NaN values: {nan_count}")
            log_subheader(f"  Inf values: {inf_count}")
            log_subheader(f"  Zero values: {zero_count} ({zero_count/board_emb.numel()*100:.1f}%)")
        
        # Move embeddings detailed analysis
        if sample['move_embeddings'] is not None:
            move_embs = sample['move_embeddings']
            log_subheader(f"\nMove embeddings analysis:")
            log_subheader(f"  Shape: {move_embs.shape}")
            log_subheader(f"  Data type: {move_embs.dtype}")
            log_subheader(f"  Device: {move_embs.device}")
            log_subheader(f"  Number of legal moves: {move_embs.shape[0]}")
            log_subheader(f"  Embedding dimension: {move_embs.shape[1]}")
            log_subheader(f"  Overall min: {move_embs.min():.6f}")
            log_subheader(f"  Overall max: {move_embs.max():.6f}")
            log_subheader(f"  Overall mean: {move_embs.mean():.6f}")
            log_subheader(f"  Overall std: {move_embs.std():.6f}")
            
            # Check for problematic values
            nan_count = torch.isnan(move_embs).sum().item()
            inf_count = torch.isinf(move_embs).sum().item()
            zero_count = (move_embs == 0).sum().item()
            log_subheader(f"  NaN values: {nan_count}")
            log_subheader(f"  Inf values: {inf_count}")
            log_subheader(f"  Zero values: {zero_count} ({zero_count/move_embs.numel()*100:.1f}%)")
            
            # Analyze each move embedding
            log_subheader(f"\nIndividual move analysis:")
            for move_idx in range(min(5, move_embs.shape[0])):  # Show first 5 moves
                move_emb = move_embs[move_idx]
                is_chosen = move_idx == sample['chosen_move_idx']
                status = " (CHOSEN)" if is_chosen else ""
                log_subheader(f"  Move {move_idx}{status}:")
                log_subheader(f"    Min: {move_emb.min():.6f}, Max: {move_emb.max():.6f}, Mean: {move_emb.mean():.6f}")
                log_subheader(f"    First 5 values: {move_emb[:5].tolist()}")
            
            # Special analysis for chosen move
            if 0 <= sample['chosen_move_idx'] < move_embs.shape[0]:
                chosen_emb = move_embs[sample['chosen_move_idx']]
                log_subheader(f"\nChosen move detailed analysis:")
                log_subheader(f"  Index: {sample['chosen_move_idx']}")
                log_subheader(f"  Min: {chosen_emb.min():.6f}")
                log_subheader(f"  Max: {chosen_emb.max():.6f}")
                log_subheader(f"  Mean: {chosen_emb.mean():.6f}")
                log_subheader(f"  Std: {chosen_emb.std():.6f}")
                log_subheader(f"  Full embedding: {chosen_emb.tolist()}")
            else:
                log_subheader(f"\nERROR: Chosen move index {sample['chosen_move_idx']} is invalid!")
        
        log_header("=== END DETAILED DEBUG ===")
    
    def create_balanced_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        """
        Create balanced train/validation/test splits ensuring:
        1. No samples from the same game appear in different splits
        2. Balanced distribution of game phases across splits
        3. Proper randomization with reproducible results
        
        Args:
            train_ratio: Proportion for training set (default: 0.8)
            val_ratio: Proportion for validation set (default: 0.1) 
            test_ratio: Proportion for test set (default: 0.1)
            random_seed: Random seed for reproducibility
            
        Returns:
            dict: Contains train_indices, val_indices, test_indices
        """
        import random
        import numpy as np
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        log_header("Creating balanced train/validation/test splits")
        
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Group samples by game file to ensure no game appears in multiple splits
        game_groups = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            game_file = sample['game_file']
            game_groups[game_file].append(idx)
        
        # Calculate statistics for each game
        game_stats = {}
        for game_file, indices in game_groups.items():
            samples = [self.samples[idx] for idx in indices]
            phase_counts = Counter(s['game_phase'] for s in samples)
            game_stats[game_file] = {
                'indices': indices,
                'total_samples': len(indices),
                'phase_counts': phase_counts,
                'phases': list(phase_counts.keys())
            }
        
        log_subheader(f"Dataset contains {len(game_groups)} games with {len(self.samples)} total samples")
        
        # Calculate overall phase distribution
        overall_phase_counts = Counter()
        for sample in self.samples:
            overall_phase_counts[sample['game_phase']] += 1
        
        log_subheader("Overall phase distribution:")
        for phase, count in overall_phase_counts.items():
            log_subheader(f"  {phase}: {count} samples ({count/len(self.samples)*100:.1f}%)")
        
        # Sort games by total samples (process larger games first for better balancing)
        sorted_games = sorted(game_stats.items(), key=lambda x: x[1]['total_samples'], reverse=True)
        
        # Initialize splits
        train_games = []
        val_games = []
        test_games = []
        
        train_samples = 0
        val_samples = 0
        test_samples = 0
        
        train_phase_counts = Counter()
        val_phase_counts = Counter()
        test_phase_counts = Counter()
        
        # Target sample counts for each split
        total_samples = len(self.samples)
        target_train = int(total_samples * train_ratio)
        target_val = int(total_samples * val_ratio)
        target_test = total_samples - target_train - target_val  # Ensure exact total
        
        log_subheader(f"Target distribution: train={target_train}, val={target_val}, test={target_test}")
        
        # Assign games to splits using a greedy algorithm that balances both size and phase distribution
        for game_file, stats in sorted_games:
            game_samples = stats['total_samples']
            game_phase_counts = stats['phase_counts']
            
            # Calculate current split ratios
            current_total = train_samples + val_samples + test_samples
            if current_total == 0:
                # First game goes to train
                best_split = 'train'
            else:
                train_ratio_current = train_samples / current_total
                val_ratio_current = val_samples / current_total
                test_ratio_current = test_samples / current_total
                
                # Calculate how much each split needs more samples
                train_deficit = max(0, train_ratio - train_ratio_current)
                val_deficit = max(0, val_ratio - val_ratio_current) 
                test_deficit = max(0, test_ratio - test_ratio_current)
                
                # Choose split with highest deficit, but also consider absolute counts
                train_score = train_deficit * 2 + max(0, (target_train - train_samples) / target_train)
                val_score = val_deficit * 2 + max(0, (target_val - val_samples) / max(1, target_val))
                test_score = test_deficit * 2 + max(0, (target_test - test_samples) / max(1, target_test))
                
                # Don't overflow splits significantly
                if train_samples + game_samples > target_train * 1.1:
                    train_score *= 0.1
                if val_samples + game_samples > target_val * 1.5:  # More lenient for smaller splits
                    val_score *= 0.1
                if test_samples + game_samples > target_test * 1.5:
                    test_score *= 0.1
                
                scores = {'train': train_score, 'val': val_score, 'test': test_score}
                best_split = max(scores.items(), key=lambda x: x[1])[0]
            
            # Assign game to the chosen split
            if best_split == 'train':
                train_games.append(game_file)
                train_samples += game_samples
                train_phase_counts.update(game_phase_counts)
            elif best_split == 'val':
                val_games.append(game_file)
                val_samples += game_samples
                val_phase_counts.update(game_phase_counts)
            else:  # test
                test_games.append(game_file)
                test_samples += game_samples
                test_phase_counts.update(game_phase_counts)
        
        # Create index mappings
        train_indices = []
        val_indices = []
        test_indices = []
        
        for game_file in train_games:
            train_indices.extend(game_stats[game_file]['indices'])
        for game_file in val_games:
            val_indices.extend(game_stats[game_file]['indices'])
        for game_file in test_games:
            test_indices.extend(game_stats[game_file]['indices'])
        
        # Shuffle indices within each split for better randomization
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)
        
        # Print final statistics
        log_subheader("\nFinal split distribution:")
        log_subheader(f"Train: {len(train_games)} games, {len(train_indices)} samples ({len(train_indices)/len(self.samples)*100:.1f}%)")
        log_subheader(f"Val:   {len(val_games)} games, {len(val_indices)} samples ({len(val_indices)/len(self.samples)*100:.1f}%)")
        log_subheader(f"Test:  {len(test_games)} games, {len(test_indices)} samples ({len(test_indices)/len(self.samples)*100:.1f}%)")
        
        log_subheader("\nPhase distribution in each split:")
        for phase in overall_phase_counts.keys():
            train_count = train_phase_counts.get(phase, 0)
            val_count = val_phase_counts.get(phase, 0)
            test_count = test_phase_counts.get(phase, 0)
            total_phase = overall_phase_counts[phase]
            
            log_subheader(f"{phase}:")
            log_subheader(f"  Train: {train_count}/{total_phase} ({train_count/total_phase*100:.1f}%)")
            log_subheader(f"  Val:   {val_count}/{total_phase} ({val_count/total_phase*100:.1f}%)")
            log_subheader(f"  Test:  {test_count}/{total_phase} ({test_count/total_phase*100:.1f}%)")
        
        # Verify no overlap
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        assert len(train_set & val_set) == 0, "Train and validation sets overlap!"
        assert len(train_set & test_set) == 0, "Train and test sets overlap!"
        assert len(val_set & test_set) == 0, "Validation and test sets overlap!"
        assert len(train_set) + len(val_set) + len(test_set) == len(self.samples), "Split indices don't cover all samples!"
        
        log_subheader("✓ Split validation passed - no overlaps detected")
        
        splits = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_games': train_games,
            'val_games': val_games,
            'test_games': test_games,
            'split_stats': {
                'train': {'samples': len(train_indices), 'games': len(train_games), 'phase_counts': dict(train_phase_counts)},
                'val': {'samples': len(val_indices), 'games': len(val_games), 'phase_counts': dict(val_phase_counts)},
                'test': {'samples': len(test_indices), 'games': len(test_games), 'phase_counts': dict(test_phase_counts)}
            }
        }
        
        return splits
    
    def save_splits(self, splits, output_dir: str):
        """
        Save the train/validation/test splits to disk
        
        Args:
            splits: Output from create_balanced_splits()
            output_dir: Directory to save split files
        """
        import os
        import pickle
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        log_header("Saving train/validation/test splits")
        
        # Save split indices
        splits_path = os.path.join(output_dir, 'dataset_splits.pkl')
        with open(splits_path, 'wb') as f:
            pickle.dump(splits, f)
        log_subheader(f"Saved split indices to {splits_path}")
        
        # Save human-readable split information
        split_info_path = os.path.join(output_dir, 'split_info.json')
        split_info = {
            'total_samples': len(self.samples),
            'total_games': len(set(s['game_file'] for s in self.samples)),
            'splits': splits['split_stats']
        }
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        log_subheader(f"Saved split information to {split_info_path}")
        
        # Save individual split files for easy loading
        for split_name, indices in [('train', splits['train_indices']), 
                                   ('val', splits['val_indices']), 
                                   ('test', splits['test_indices'])]:
            split_file = os.path.join(output_dir, f'{split_name}_indices.npy')
            np.save(split_file, np.array(indices))
            log_subheader(f"Saved {split_name} indices to {split_file}")
        
        # Create split-specific datasets if needed
        for split_name, indices in [('train', splits['train_indices']), 
                                   ('val', splits['val_indices']), 
                                   ('test', splits['test_indices'])]:
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Extract samples for this split
            split_samples = [self.samples[i] for i in indices]
            
            # Create a temporary dataset instance for this split
            split_dataset = type(self).__new__(type(self))
            split_dataset.samples = split_samples
            split_dataset.device = self.device
            split_dataset.gnn_model = self.gnn_model
            # Copy all necessary attributes
            split_dataset.move_embedding_mode = getattr(self, 'move_embedding_mode', 'difference')
            split_dataset.winner_only = getattr(self, 'winner_only', False)
            split_dataset.sampling_strategy = getattr(self, 'sampling_strategy', 'distributed')
            split_dataset.max_moves_per_game = getattr(self, 'max_moves_per_game', 15)
            
            # Save split-specific embeddings
            split_dataset.save_embeddings(split_dir)
            
            log_subheader(f"Saved {split_name} embeddings to {split_dir} ({len(split_samples)} samples)")
        
        log_header(f"All splits saved to {output_dir}")
        
        return splits_path


def main():
    """
    Load a GNN model (.pt), make it return embeddings from pro_matches,
    create a dataset where each sample consists of:
    [gnn_embedding of the board, list of embeddings of the legal moves obtained as: 
     (embedding of the board after the move)-(embedding of the board before the move), 
     chosen move(ground truth)]
    """
    
    parser = argparse.ArgumentParser(description="Generate LLM training dataset from GNN embeddings")
    parser.add_argument('--gnn_model', type=str, required=True, help='Path to trained GNN model (.pt file)')
    parser.add_argument('--game_dir', type=str, default='pro_matches/GNN_Apr-3-2024', help='Directory with game files')
    parser.add_argument('--output_dir', type=str, default='data/llm_dataset', help='Output directory for dataset')
    parser.add_argument('--cache_file', type=str, default=None, help='Cache file to save/load processed dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--debug_sample', type=int, default=None, help='Debug specific sample by index')
    parser.add_argument('--num_debug_samples', type=int, default=3, help='Number of samples to show in general debug output')
    parser.add_argument('--sampling_strategy', type=str, default='distributed', 
                       choices=['distributed', 'all', 'random', 'unlimited'],
                       help='Strategy for sampling moves from games: distributed (opening/middle/endgame), all (sequential up to max_moves_per_game), random (random selection up to max_moves_per_game), or unlimited (ALL moves ignoring max_moves_per_game)')
    parser.add_argument('--max_moves_per_game', type=int, default=15, 
                       help='Maximum number of moves to process per game (ignored for unlimited strategy)')
    parser.add_argument('--move_embedding_mode', type=str, default='difference',
                       choices=['difference', 'concatenation'],
                       help='Mode for computing move embeddings: difference (after-before, smaller) or concatenation (concat(before,after), 2x larger)')
    parser.add_argument('--winner_only', action='store_true', default=False,
                       help='Only extract samples from winning player moves')
    parser.add_argument('--create_splits', action='store_true', default=False,
                       help='Create train/validation/test splits (80/10/10)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Proportion of data for training (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, 
                       help='Proportion of data for validation (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Proportion of data for testing (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        log_subheader("CUDA not available, using CPU")
    
    os.environ["TORCH_DEVICE"] = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create dataset
    if args.cache_file and os.path.exists(args.cache_file):
        log_header("Loading dataset from cache")
        dataset = LLMDataset.load_from_cache(args.cache_file, args.gnn_model, args.device, args.move_embedding_mode)
    else:
        log_header("Creating new dataset")
        dataset = LLMDataset(
            gnn_model_path=args.gnn_model,
            game_files_dir=args.game_dir,
            device=args.device,
            sampling_strategy=args.sampling_strategy,
            max_moves_per_game=args.max_moves_per_game,
            move_embedding_mode=args.move_embedding_mode,
            winner_only=args.winner_only
        )
        
        # Save to cache if specified
        if args.cache_file:
            dataset.save_to_cache(args.cache_file)
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    log_header("Dataset Statistics")
    for key, value in stats.items():
        log_subheader(f"{key}: {value}")
    
    # Save statistics to CSV
    csv_path = os.path.join(args.output_dir, "dataset_stats.csv")
    dataset.save_to_csv(csv_path)
    
    # Save actual embeddings data
    dataset.save_embeddings(args.output_dir)
    
    # Create train/validation/test splits if requested
    if args.create_splits:
        log_header("Creating balanced train/validation/test splits")
        splits = dataset.create_balanced_splits(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
        
        # Save splits to disk
        dataset.save_splits(splits, args.output_dir)
        
        # Print final split summary
        log_header("Split Summary")
        for split_name, split_data in splits['split_stats'].items():
            log_subheader(f"{split_name.capitalize()}:")
            log_subheader(f"  Games: {split_data['games']}")
            log_subheader(f"  Samples: {split_data['samples']}")
            log_subheader(f"  Phase distribution: {split_data['phase_counts']}")
    
    # Debug specific sample if requested
    if args.debug_sample is not None:
        log_header(f"Debugging specific sample {args.debug_sample}")
        dataset.debug_sample_content(args.debug_sample)
    
    # Create DataLoader
    dataloader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    
    # Test the dataloader
    log_header("Testing DataLoader")
    for i, batch in enumerate(dataloader):
        log_subheader(f"Batch {i+1}:")
        log_subheader(f"  Board embeddings shape: {batch['board_embedding'].shape}")
        log_subheader(f"  Move embeddings shape: {batch['move_embeddings'].shape}")
        log_subheader(f"  Next board embeddings shape: {batch['next_board_embeddings'].shape}")
        log_subheader(f"  Chosen move indices shape: {batch['chosen_move_idx'].shape}")
        log_subheader(f"  Chosen move indices sample: {batch['chosen_move_idx'][:5].tolist()}")
        
        # Print stats for first batch
        if i == 0:
            board_embs = batch['board_embedding']
            move_embs = batch['move_embeddings']
            next_board_embs = batch['next_board_embeddings']
            log_subheader(f"  Board embeddings stats: min={board_embs.min():.4f}, max={board_embs.max():.4f}, mean={board_embs.mean():.4f}")
            log_subheader(f"  Move embeddings stats: min={move_embs.min():.4f}, max={move_embs.max():.4f}, mean={move_embs.mean():.4f}")
            log_subheader(f"  Next board embeddings stats: min={next_board_embs.min():.4f}, max={next_board_embs.max():.4f}, mean={next_board_embs.mean():.4f}")
            
            # Show sample values from first item in batch
            log_subheader(f"  First board embedding sample: {board_embs[0][:5].tolist()}")
            log_subheader(f"  First move embeddings sample (first move): {move_embs[0][0][:5].tolist()}")
            log_subheader(f"  First next board embeddings sample (first move): {next_board_embs[0][0][:5].tolist()}")
        
        if i >= 2:  # Only show first 3 batches
            break
    
    log_header("Dataset creation completed successfully!")
    
    return dataset


if __name__ == "__main__":
    main()
