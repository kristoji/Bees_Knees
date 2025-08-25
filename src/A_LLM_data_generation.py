'''
Key features:
- Batch GNN inference for all legal moves at a position (drastically fewer forward() calls).
- Avoid redundant embedding computation (reuse board_embedding_before; reuse chosen next_board_embedding).
- Optional quiet mode to reduce logging overhead (default).
- Optional skip of dataloader smoke test (which can be slow) unless explicitly requested.
- Minor micro-optimizations in chosen-move matching.
- Creates ONE complete sequence per game containing all moves - no overlapping sequences, no multiple samples per game.
'''

import os
import re
import argparse
import pickle
from glob import glob
from typing import List, Tuple, Optional, Dict, Iterable, Union

import numpy as np
import torch


from torch_geometric.data import Batch


from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ai.oracleGNN import OracleGNN
from ai.oracleRND import OracleRND
from ai.log_utils import reset_log, log_header, log_subheader
from engine.board import Board
from engine.game import Move
from engine.enums import GameState


class SequentialGameParser:
    """Parse game board files to extract sequential moves and game states for historical context"""

    @staticmethod
    def parse_game_file(file_path: str) -> Tuple[str, List[str], str]:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        parts = content.split(';')
        if len(parts) < 3:
            raise ValueError(f"Invalid game format in {file_path}")

        game_type = parts[0]
        game_state = parts[1]
        moves = parts[3:] if len(parts) > 3 else []

        winner = "Unknown"
        if "BlackWins" in game_state:
            winner = "Black"
        elif "WhiteWins" in game_state:
            winner = "White"

        initial_gamestring = f"{game_type};NotStarted;White[1]"
        return initial_gamestring, moves, winner

    @staticmethod
    def create_board_from_gamestring(gamestring: str) -> Board:
        return Board(gamestring)


class SequentialLLMDataset(Dataset):
    def __init__(
        self,
        gnn_model_path: str,
        game_files_dir: str,
        device: str = "cuda",
        sequence_length: int = 6,
        max_moves_per_game: Optional[int] = None,
        move_embedding_mode: str = "difference",
        winner_only: bool = False,
        output_dir: Optional[str] = None,
        verbose: bool = False,
        max_games: Optional[int] = None,
        store_legal_move_embeddings: bool = False,
    ):
        """Initialize sequential dataset generator."""
        self.device = device
        self.game_files_dir = game_files_dir
        self.max_games = max_games
        self.sequence_length = sequence_length
        self.max_moves_per_game = max_moves_per_game
        self.move_embedding_mode = move_embedding_mode
        self.winner_only = winner_only
        self.output_dir = output_dir
        self.verbose = verbose
        self.store_legal_move_embeddings = store_legal_move_embeddings

        # Load the GNN model
        if self.verbose:
            log_header("Loading GNN model")
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
        self.gnn_model = OracleGNN(device=str(device), hidden_dim=256, **kwargs_network)
        self.gnn_model.load(gnn_model_path)
        self.gnn_model.network.eval()

        # Discover all game files
        all_game_files = glob(os.path.join(game_files_dir, "board*.txt"))
        all_game_files = sorted(all_game_files)
        original_count = len(all_game_files)
        # Optionally limit how many games to process
        if self.max_games is not None and self.max_games > 0:
            self.game_files = all_game_files[: self.max_games]
            log_subheader(f"Found {original_count} game files; limiting to first {len(self.game_files)} based on max_games={self.max_games}")
        else:
            self.game_files = all_game_files
            log_subheader(f"Found {len(self.game_files)} game files")

        # Process
        self.samples = []
        self._process_all_games()

        if self.verbose:
            log_subheader(f"Created {len(self.samples)} sequential training samples")
            self._debug_print_samples(num_samples=3)

    # ----------------------
    # Core embedding helpers
    # ----------------------
    @torch.inference_mode()
    def _get_board_embedding(self, board: Board) -> Optional[torch.Tensor]:
        """Return shape: [1, gin_dim] on success, else None"""
        try:
            data = self.gnn_model._data_from_board(board)
            if data is None:
                return None
            data = data.to(self.device)
            emb = self.gnn_model.network.return_embedding(data)
            return emb.detach().cpu()
        except Exception:
            return None

    @torch.inference_mode()
    def _batch_move_embeddings(
        self,
        board: Board,
        valid_moves: List[Move],
        board_embedding_before: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Compute embeddings for all legal moves in *one* or few batched forward passes.
        Returns:
            legal_move_embeddings: [num_moves, 1, move_dim]
            legal_move_texts: list[str]
            next_board_embeddings: [num_moves, 1, gin_dim]
        """
        move_texts: List[str] = []
        data_list = []
        # Create list of "after move" board graphs
        for mv in valid_moves:
            try:
                move_texts.append(board.stringify_move(mv))
                board.safe_play(mv)
                data = self.gnn_model._data_from_board(board)
                board.undo()
                if data is not None:
                    data_list.append(data)
                else:
                    data_list.append(None)
            except Exception:
                move_texts.append("ERROR_MOVE")
                data_list.append(None)

        # If PyG Batch is available, run a single forward on the non-None subset, then rebuild order
        next_embeds_list: List[Optional[torch.Tensor]] = [None] * len(data_list)
        # Indices that are actually valid
        valid_idx = [i for i, d in enumerate(data_list) if d is not None]
        if valid_idx:
            batch = Batch.from_data_list([data_list[i] for i in valid_idx]).to(self.device)
            after_emb = self.gnn_model.network.return_embedding(batch).detach().cpu()
            # Split by batch using 'batch' vector produced by PyG
            # Some models already return per-graph embeddings; if so, the length matches len(valid_idx).
            # We handle both single-graph and batched returns.
            if after_emb.dim() == 1:
                after_emb = after_emb.unsqueeze(0)
            # Assign back
            for j, i in enumerate(valid_idx):
                next_embeds_list[i] = after_emb[j].unsqueeze(0)  # [1, gin_dim]

        # Fallback path or fill missing with zeros by per-move forward
        for i, d in enumerate(data_list):
            if next_embeds_list[i] is None:
                if d is None:
                    # Could not build graph; use zeros
                    zero = torch.zeros_like(board_embedding_before)
                    next_embeds_list[i] = zero
                else:
                    dd = d.to(self.device)
                    emb = self.gnn_model.network.return_embedding(dd).detach().cpu()
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)
                    next_embeds_list[i] = emb  # [1, gin_dim]

        next_board_embeddings = torch.stack(next_embeds_list, dim=0)  # [num_moves, 1, gin_dim]

        # Build move embeddings
        if self.move_embedding_mode == "concatenation":
            board_before_tiled = board_embedding_before.expand(next_board_embeddings.size(0), -1, -1)  # [num_moves, 1, gin_dim]
            move_embeddings = torch.cat([board_before_tiled, next_board_embeddings], dim=-1)  # [num_moves, 1, 2*gin_dim]
        else:  # "difference"
            move_embeddings = next_board_embeddings - board_embedding_before  # [num_moves, 1, gin_dim]

        return move_embeddings, move_texts, next_board_embeddings

    def _find_chosen_move_index(self, board: Board, move_str: str, valid_moves: List[Move]) -> Optional[int]:
        """Find index of chosen move in valid_moves by string match first; O(n)."""
        try:
            chosen_move = board._parse_move(move_str)
            if chosen_move is None:
                return None
            chosen_move_str = board.stringify_move(chosen_move)
            for idx, valid_move in enumerate(valid_moves):
                if chosen_move_str == board.stringify_move(valid_move):
                    return idx
            # Slow path: field-wise comparison as fallback
            for idx, valid_move in enumerate(valid_moves):
                if (chosen_move.bug == valid_move.bug and
                    chosen_move.origin == valid_move.origin and
                    chosen_move.destination == valid_move.destination):
                    return idx
            return None
        except Exception:
            return None

    def _process_all_games(self):
        parser = SequentialGameParser()
        total_sequences_created = 0

        for game_file in tqdm(self.game_files, desc="Processing games"):
            try:
                initial_gamestring, moves, winner = parser.parse_game_file(game_file)
                board = parser.create_board_from_gamestring(initial_gamestring)

                board_sequence = []  # list[Tensor 1xD]
                move_sequence = []   # list[Tensor 1x(D or 2D)]
                next_board_sequence = []  # list[Tensor 1xD]
                move_strings_sequence = []
                chosen_indices_sequence = []
                legal_move_counts_sequence = []
                # Optional storage of full legal move embeddings (ragged list of tensors)
                if self.store_legal_move_embeddings:
                    legal_move_embeddings_sequence: List[torch.Tensor] = []
                    # store the corresponding next-board embeddings for every legal move
                    legal_next_board_embeddings_sequence: List[torch.Tensor] = []

                # Iterate through moves - process ALL moves without limit
                for move_idx, move_str in enumerate(moves):
                    # winner_only filter (still advance board to keep state consistent)
                    if self.winner_only:
                        current_player = "White" if move_idx % 2 == 0 else "Black"
                        if current_player != winner:
                            try:
                                board.play(move_str)
                                continue
                            except Exception as e:
                                if self.verbose:
                                    log_subheader(f"Error playing move {move_str}: {e}")
                                break

                    # Skip / advance pass moves
                    if move_str.strip() == "pass":
                        try:
                            board.play(move_str)
                            continue
                        except Exception as e:
                            if self.verbose:
                                log_subheader(f"Error playing PASS: {e}")
                            break

                    # End if game is already finished
                    if board.state not in (GameState.IN_PROGRESS, GameState.NOT_STARTED):
                        break

                    # Compute current board embedding ONCE
                    board_embedding_before = self._get_board_embedding(board)
                    if board_embedding_before is None:
                        # Advance and continue
                        try:
                            board.play(move_str)
                        except Exception:
                            pass
                        continue

                    # Legal moves
                    valid_moves = list(board.get_valid_moves())
                    if not valid_moves:
                        try:
                            board.play(move_str)
                        except Exception:
                            pass
                        continue

                    # Batched embeddings for all legal moves (and their next states)
                    move_result = self._batch_move_embeddings(board, valid_moves, board_embedding_before)
                    legal_move_embeddings, legal_move_texts, next_board_embeddings = move_result

                    # Which move was chosen?
                    chosen_move_idx = self._find_chosen_move_index(board, move_str, valid_moves)
                    if chosen_move_idx is None:
                        # Could not match; still advance board
                        try:
                            board.play(move_str)
                        except Exception:
                            pass
                        continue

                    # Advance the real board state
                    try:
                        board.play(move_str)
                    except Exception as e:
                        if self.verbose:
                            log_subheader(f"Error playing move {move_str}: {e}")
                        break

                    # Reuse the precomputed next embedding for the chosen move (no extra GNN call)
                    board_embedding_after = next_board_embeddings[chosen_move_idx]  # [1, gin_dim]
                    chosen_move_embedding = legal_move_embeddings[chosen_move_idx]  # [1, move_dim]

                    # Append sequences
                    board_sequence.append(board_embedding_before)
                    move_sequence.append(chosen_move_embedding)
                    next_board_sequence.append(board_embedding_after)
                    move_strings_sequence.append(move_str)
                    chosen_indices_sequence.append(int(chosen_move_idx))
                    legal_move_counts_sequence.append(int(legal_move_embeddings.shape[0]))

                    # Store full legal move embeddings if requested
                    if self.store_legal_move_embeddings:
                        # detach & cpu already; just append (shape [num_legal,1,move_dim])
                        legal_move_embeddings_sequence.append(legal_move_embeddings.clone())
                        # also store the next-board embeddings for each legal move (shape [num_legal,1,gin_dim])
                        legal_next_board_embeddings_sequence.append(next_board_embeddings.clone())

                    # Continue collecting all moves for this game - will create single sample at the end

                    # Continue collecting all moves for this game - will create single sample at the end

                # Create ONE sample per game after processing all moves
                if board_sequence:
                    # Create a single sample that contains the entire game sequence
                    sample = {
                        'game_file': os.path.basename(game_file),
                        'winner': winner,
                        'total_game_moves': len(moves),
                        'processed_moves': len(board_sequence),
                        'move_strings': move_strings_sequence,
                        'chosen_move_indices': chosen_indices_sequence,
                        'legal_move_counts': legal_move_counts_sequence,
                        'board_embeddings_sequence': torch.stack(board_sequence),      # [num_moves, 1, gin_dim]
                        'chosen_move_embeddings_sequence': torch.stack(move_sequence), # [num_moves, 1, move_dim]
                        'next_board_embeddings_sequence': torch.stack(next_board_sequence), # [num_moves, 1, gin_dim]
                    }
                    if self.store_legal_move_embeddings:
                        sample['legal_move_embeddings_sequence'] = legal_move_embeddings_sequence  # ragged list of tensors
                        sample['legal_next_board_embeddings_sequence'] = legal_next_board_embeddings_sequence
                    self.samples.append(sample)
                    total_sequences_created += 1

                # Save per-game full sequence (if any steps recorded)
                if board_sequence and (self.output_dir is not None or self.game_files_dir is not None):
                    try:
                        base_out = self.output_dir if self.output_dir is not None else self.game_files_dir
                        full_seq_dir = os.path.join(base_out, "full_game_sequences")
                        os.makedirs(full_seq_dir, exist_ok=True)
                        game_base = os.path.splitext(os.path.basename(game_file))[0]
                        out_path = os.path.join(full_seq_dir, f"{game_base}.pkl")

                        full_sequence = {
                            'game_file': os.path.basename(game_file),
                            'winner': winner,
                            'num_steps': len(board_sequence),
                            'move_strings': move_strings_sequence,
                            'chosen_move_indices': chosen_indices_sequence,
                            'legal_move_counts': legal_move_counts_sequence,
                            'board_embeddings_before': torch.stack(board_sequence),
                            'chosen_move_embeddings': torch.stack(move_sequence),
                            'next_board_embeddings': torch.stack(next_board_sequence),
                        }
                        if self.store_legal_move_embeddings:
                            full_sequence['legal_move_embeddings_sequence'] = legal_move_embeddings_sequence
                            full_sequence['legal_next_board_embeddings_sequence'] = legal_next_board_embeddings_sequence

                        with open(out_path, 'wb') as f:
                            pickle.dump(full_sequence, f)
                    except Exception as e:
                        if self.verbose:
                            log_subheader(f"Failed to save full-game sequence for {game_file}: {e}")

            except Exception as e:
                if self.verbose:
                    log_subheader(f"Error processing {game_file}: {e}")
                continue

        log_header("=== SEQUENTIAL PROCESSING SUMMARY ===")
        log_subheader(f"Total game sequences created: {total_sequences_created}")
        log_subheader(f"Each sequence represents one complete game")
        log_header("=== END SUMMARY ===")

    def _debug_print_samples(self, num_samples: int = 3):
        if not self.samples:
            log_subheader("No samples to debug - dataset is empty")
            return
        log_header("=== SEQUENTIAL DATASET DEBUGGING INFO ===")
        total_samples = len(self.samples)
        log_subheader(f"Total game sequences in dataset: {total_samples}")
        
        if total_samples > 0:
            first_sample = self.samples[0]
            board_seq = first_sample['board_embeddings_sequence']
            move_seq = first_sample['chosen_move_embeddings_sequence']
            log_subheader(f"Example game sequence shapes:")
            log_subheader(f"Board embeddings: {board_seq.shape}")
            log_subheader(f"Move embeddings: {move_seq.shape}")

        num_to_show = min(num_samples, total_samples)
        log_subheader(f"Showing details for first {num_to_show} game sequences:")
        for i in range(num_to_show):
            s = self.samples[i]
            log_subheader(f"\n--- Game Sequence {i+1} ---")
            log_subheader(f"Game file: {s['game_file']}")
            log_subheader(f"Winner: {s['winner']}")
            log_subheader(f"Total moves in game: {s['total_game_moves']}")
            log_subheader(f"Processed moves: {s['processed_moves']}")
            board_emb = s['board_embeddings_sequence']
            move_emb = s['chosen_move_embeddings_sequence']
            log_subheader(f"Board embeddings stats: mean={board_emb.mean():.4f}, std={board_emb.std():.4f}")
            log_subheader(f"Move embeddings stats: mean={move_emb.mean():.4f}, std={move_emb.std():.4f}")
        log_header("=== END SEQUENTIAL DEBUGGING INFO ===")

    # Standard Dataset API
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'game_file': s['game_file'],
            'winner': s['winner'],
            'total_game_moves': s['total_game_moves'],
            'processed_moves': s['processed_moves'],
            'move_strings': s['move_strings'],
            'chosen_move_indices': s['chosen_move_indices'],
            'legal_move_counts': s['legal_move_counts'],
            'board_embeddings_sequence': s['board_embeddings_sequence'],
            'chosen_move_embeddings_sequence': s['chosen_move_embeddings_sequence'],
            'next_board_embeddings_sequence': s['next_board_embeddings_sequence'],
            'legal_move_embeddings_sequence': s.get('legal_move_embeddings_sequence'),
        }

    def save_to_cache(self, cache_path: str):
        log_header(f"Saving sequential dataset to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.samples, f)
        log_subheader("Sequential cache saved successfully")

    @classmethod
    def load_from_cache(cls, cache_path: str, gnn_model_path: str, device: str = "cuda", move_embedding_mode: str = "difference", verbose: bool = False):
        log_header(f"Loading sequential dataset from cache: {cache_path}")
        instance = cls.__new__(cls)
        instance.device = device
        instance.move_embedding_mode = move_embedding_mode
        instance.verbose = verbose

        kwargs_network = {
            'conv_type': 'GIN',
            'num_layers': 2,
            'gat_heads': 8,
            'gat_concat': True,
            'conv_dropout': 0.1,
            'mlp_dropout': 0.1,
            'final_dropout': 0.2,
            'use_batch_norm': False,
            'use_layer_norm': True,
            'use_residual': False,
            'pooling': 'add',
            'mlp_layers': 2,
            'final_mlp_layers': 2
        }
        instance.gnn_model = OracleGNN(device=device, hidden_dim=24, **kwargs_network)
        instance.gnn_model.load(gnn_model_path)
        instance.gnn_model.network.eval()

        with open(cache_path, 'rb') as f:
            instance.samples = pickle.load(f)

        log_subheader(f"Loaded {len(instance.samples)} sequential samples from cache")
        if instance.verbose:
            instance._debug_print_samples(num_samples=3)
        return instance

    @staticmethod
    def collate_fn(batch):
        game_files = [b['game_file'] for b in batch]
        winners = [b['winner'] for b in batch]
        total_game_moves = [b['total_game_moves'] for b in batch]
        processed_moves = [b['processed_moves'] for b in batch]
        move_strings = [b['move_strings'] for b in batch]
        chosen_move_indices = [b['chosen_move_indices'] for b in batch]
        legal_move_counts = [b['legal_move_counts'] for b in batch]
        board_embeddings_sequences = [b['board_embeddings_sequence'] for b in batch]
        chosen_move_embeddings_sequences = [b['chosen_move_embeddings_sequence'] for b in batch]
        next_board_embeddings_sequences = [b['next_board_embeddings_sequence'] for b in batch]
        legal_move_embeddings_sequences = [b.get('legal_move_embeddings_sequence') for b in batch]

        return {
            'game_files': game_files,
            'winners': winners,
            'total_game_moves': total_game_moves,
            'processed_moves': processed_moves,
            'move_strings': move_strings,
            'chosen_move_indices': chosen_move_indices,
            'legal_move_counts': legal_move_counts,
            'board_embeddings_sequences': board_embeddings_sequences,
            'chosen_move_embeddings_sequences': chosen_move_embeddings_sequences,
            'next_board_embeddings_sequences': next_board_embeddings_sequences,
            'legal_move_embeddings_sequences': legal_move_embeddings_sequences,
        }       


def main():
    parser = argparse.ArgumentParser(description="Generate sequential LLM training dataset (optimized)")
    parser.add_argument('--gnn_model', type =str, required=True, help='Path to trained GNN model (.pt file)')
    parser.add_argument('--game_dir', type=str, default='pro_matches/board_data_tournament', help='Directory with game files')
    parser.add_argument('--output_dir', type=str, default='data/sequential_hive_llm_dataset', help='Output directory')
    parser.add_argument('--cache_file', type=str, default=None, help='Cache file to save/load processed dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--sequence_length', type=int, default=5, help='Length of historical context sequence')
    parser.add_argument('--max_moves_per_game', type=int, default=None, help='Maximum moves to process per game (None for unlimited)')
    parser.add_argument('--move_embedding_mode', type=str, default='difference', choices=['difference','concatenation'])
    parser.add_argument('--winner_only', action='store_true', default=False, help='Only extract samples from winning player moves')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--smoke_test', action='store_true', default=False, help='Run a quick DataLoader smoke test at the end')
    parser.add_argument('--max_games', type=int, default=None, help='Maximum number of game files to process from game_dir')
    parser.add_argument('--store_legal_move_embeddings', action='store_true', default=False, help='Store full legal move embeddings and next-board embeddings per step')

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
        log_header("Loading sequential dataset from cache")
        dataset = SequentialLLMDataset.load_from_cache(
            args.cache_file, args.gnn_model, args.device, args.move_embedding_mode, verbose=args.verbose
        )
    else:
        log_header("Creating new sequential dataset (optimized)")
        dataset = SequentialLLMDataset(
            gnn_model_path=args.gnn_model,
            game_files_dir=args.game_dir,
            device=args.device,
            sequence_length=args.sequence_length,
            max_moves_per_game=args.max_moves_per_game,
            move_embedding_mode=args.move_embedding_mode,
            winner_only=args.winner_only,
            output_dir=args.output_dir,
            verbose=args.verbose,
            max_games=args.max_games,
            store_legal_move_embeddings=args.store_legal_move_embeddings,
        )
        if args.cache_file:
            dataset.save_to_cache(args.cache_file)

    # Group-aware train/val split by game file (deterministic)
    log_header("Creating group-aware train/validation splits (by game file)")
    from collections import defaultdict
    
    # Each sample represents one complete game, so we split by sample directly
    game_files = [s['game_file'] for s in dataset.samples]
    unique_games = list(set(game_files))
    
    rng = np.random.default_rng(42)
    rng.shuffle(unique_games)
    split_idx = int(len(unique_games) * 0.8)
    train_games = set(unique_games[:split_idx])
    val_games = set(unique_games[split_idx:])

    train_samples = [s for s in dataset.samples if s['game_file'] in train_games]
    val_samples = [s for s in dataset.samples if s['game_file'] in val_games]

    train_cache_path = os.path.join(args.output_dir, "train_sequential_cache.pkl")
    val_cache_path   = os.path.join(args.output_dir, "validation_sequential_cache.pkl")
    with open(train_cache_path, 'wb') as f:
        pickle.dump(train_samples, f)
    with open(val_cache_path, 'wb') as f:
        pickle.dump(val_samples, f)
    log_subheader(f"Saved {len(train_samples)} training samples to {train_cache_path}")
    log_subheader(f"Saved {len(val_samples)} validation samples to {val_cache_path}")

    # Optional: quick smoke test of DataLoader
    if args.smoke_test:
        log_header("Testing Sequential DataLoader (smoke test)")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
        for i, batch in enumerate(dataloader):
            log_subheader(f"Batch {i+1}: "
                          f"games={len(batch['game_files'])}, "
                          f"total_moves={batch['total_game_moves']}, "
                          f"processed_moves={batch['processed_moves']}")
            for j in range(min(2, len(batch['game_files']))):
                log_subheader(f"  Game {j}: {batch['game_files'][j]}, "
                              f"winner={batch['winners'][j]}, "
                              f"moves={batch['processed_moves'][j]}, "
                              f"board_seq_shape={batch['board_embeddings_sequences'][j].shape}")
            if i >= 2:
                break
        log_header("Smoke test completed.")

    log_header("Sequential dataset creation completed successfully!")
    return dataset


if __name__ == "__main__":
    main()
