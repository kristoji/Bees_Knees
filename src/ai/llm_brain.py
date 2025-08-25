import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch

from engine.board import Board
from engine.game import Move
from engine.enums import GameState
from ai.brains import Brain
from ai.oracleGNN import OracleGNN

# Reuse the sequential player used during training/inference
from A_LLM_trainer import (
    SequentialHiveLLMPlayer,
    SequentialHiveLLMConfig,
)


class LLMSequentialBrain(Brain):
    """
    Brain that uses the finetuned Sequential HIVE LLM to choose moves via
    embedding-space generation + nearest-legal matching.

    It keeps a running history of (board_before, move_embedding) pairs, using
    'difference' mode for move embeddings: move_emb = next_board - before_board.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        gnn_model_path: str = os.path.join("src", "models", "pretrain_GIN_3.pt"),
        sequence_length: int = 5,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Load config (defaults are fine if no saved config JSON used here)
        # Always train/infer with exactly 5 examples to mirror finetuning prompt
        self.cfg = SequentialHiveLLMConfig(sequence_length=5)
        # Ensure we use the placeholder soft-prompt path (not free generation)
        self.cfg.use_free_generation = False

        # Choose model directory (prefer sequential_hive_llm_v1, else fallback)
        if model_dir is None:
            cand = [
                os.path.join("models", "LLM_1000_tournament_mse", "best_model"),
            ]
            model_dir = next((p for p in cand if os.path.isdir(p)), cand[0])
        self.model_dir = model_dir

        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load LLM player (quantized if available) and adapter
        self.player = SequentialHiveLLMPlayer(self.cfg)
        # Try to load adapter weights saved at training time
        adapter_path = os.path.join(self.model_dir, "lora_adapter")
        try:
            # Some architectures expose this API when wrapped by PEFT
            self.player.base_model.load_adapter(adapter_path, adapter_name="trained")  # type: ignore[attr-defined]
            self.player.base_model.set_adapter("trained")  # type: ignore[attr-defined]
        except Exception:
            # Fallback: try to load as PEFT checkpoint directly
            try:
                from peft import PeftModel

                self.player.base_model = PeftModel.from_pretrained(
                    self.player.base_model, adapter_path
                )
            except Exception:
                # If adapters cannot be loaded, proceed with base weights
                pass

        # Ensure projection bridge follows the base model's device
        base_device = next(self.player.base_model.parameters()).device
        self.player.projection_module.to(base_device)
        self.player.eval()

        # GNN for board embeddings
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
            'final_mlp_layers': 3,
        }
        # Important: hidden_dim must match cfg.gin_embedding_dim
        self.oracle = OracleGNN(device=str(self.device), hidden_dim=self.cfg.gin_embedding_dim, **kwargs_network)
        self.oracle.load(gnn_model_path)
        self.oracle.network.eval()

        # Histories: store last K pairs - now tracking full game history
        # Force the rolling context length to 5 examples
        self.seq_len = 5
        self.board_hist = deque(maxlen=self.seq_len)
        self.move_hist = deque(maxlen=self.seq_len)

        # Track the actual game moves for the live game (for building training-style prompt)
        # All board states and moves in game order
        self.game_board_embeddings = []
        self.game_move_embeddings = []

        # Track last board embedding after our own last move (to infer opponent move diff)
        self._last_board_after_our_move = None

        # Track game state to know when to reset
        self._last_board_state = None

    # --- Embedding helpers ---
    @torch.no_grad()
    def _embed_board(self, board: Board) -> Optional[torch.Tensor]:
        d = self.oracle._data_from_board(board)
        if d is None:
            return None
        d = d.to(self.device)
        emb = self.oracle.network.return_embedding(d)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)  # [1, gin]
        return emb.detach().cpu()

    @torch.no_grad()
    def _batch_after_embeddings(self, board: Board, moves: List[Move]) -> List[torch.Tensor]:
        """Return list of [1, gin] embeddings for each board after applying moves."""
        try:
            from torch_geometric.data import Batch
        except Exception:
            # Fallback to per-move embedding if PyG Batch is not available
            outs: List[torch.Tensor] = []
            for mv in moves:
                board.safe_play(mv)
                d = self.oracle._data_from_board(board)
                board.undo()
                if d is None:
                    outs.append(torch.zeros(1, self.cfg.gin_embedding_dim))
                else:
                    dd = d.to(self.device)
                    e = self.oracle.network.return_embedding(dd)
                    if e.dim() == 1:
                        e = e.unsqueeze(0)
                    outs.append(e.detach().cpu())
            return outs

        data_list = []
        for mv in moves:
            board.safe_play(mv)
            d = self.oracle._data_from_board(board)
            board.undo()
            data_list.append(d)

        # Batch only valid graphs
        idx_map = [i for i, d in enumerate(data_list) if d is not None]
        outs: List[Optional[torch.Tensor]] = [None] * len(data_list)
        if idx_map:
            batch = Batch.from_data_list([data_list[i] for i in idx_map]).to(self.device)
            emb = self.oracle.network.return_embedding(batch).detach().cpu()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            for j, i in enumerate(idx_map):
                outs[i] = emb[j].unsqueeze(0)
        # Fill missing with zeros
        zero = torch.zeros(1, self.cfg.gin_embedding_dim)
        return [x if x is not None else zero for x in outs]

    def _reset_game_history(self):
        """Reset game history when starting a new game"""
        self.game_board_embeddings.clear()
        self.game_move_embeddings.clear()
        self.board_hist.clear()
        self.move_hist.clear()
        self._last_board_after_our_move = None
        self._last_board_state = None

    def _build_training_style_prompt(self, current_board_embedding: torch.Tensor, 
                                   legal_move_embeddings: torch.Tensor,
                                   legal_move_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the exact training-style prompt format that the LLM was trained on:
        "Board i: <embedding> Move i: <embedding>" for the last sequence_length moves.
        
        Returns context embeddings in the format expected by the model.
        """
        # Normalize current board shape to [1, gin]
        if current_board_embedding.dim() == 1:
            cur = current_board_embedding.unsqueeze(0)
        elif current_board_embedding.dim() == 2:
            cur = current_board_embedding
        elif current_board_embedding.dim() == 3 and current_board_embedding.size(0) == 1:
            # e.g., [1, 1, gin] -> [1, gin]
            cur = current_board_embedding.squeeze(1)
        else:
            # Fallback: flatten extra dims except last
            cur = current_board_embedding.view(1, -1)

        # Create empty board embedding for padding: shape [1, gin]
        empty_board = torch.zeros_like(cur)
        zero_move = torch.zeros_like(cur)

        # Take only the last K examples and left-pad with zeros to reach seq_len
        history_len = min(self.seq_len, len(self.game_board_embeddings))
        boards_tail = self.game_board_embeddings[-history_len:]
        moves_tail = self.game_move_embeddings[-history_len:]

        pad_count = self.seq_len - history_len

        def _norm_1_gin(t: torch.Tensor) -> torch.Tensor:
            # Ensure tensors are [1, gin]
            if t is None:
                return empty_board.clone()
            if t.dim() == 1:
                return t.unsqueeze(0)
            if t.dim() == 2:
                return t
            if t.dim() == 3 and t.size(0) == 1:
                return t.squeeze(1)
            # Generic fallback
            return t.view(1, -1)

        seq_boards: List[torch.Tensor] = [empty_board.clone() for _ in range(pad_count)] + [
            _norm_1_gin(b) for b in boards_tail
        ]

        # Align moves with boards_tail; if missing, fill with zeros
        padded_moves_tail: List[torch.Tensor] = []
        for i in range(history_len):
            if i < len(moves_tail):
                padded_moves_tail.append(_norm_1_gin(moves_tail[i]))
            else:
                padded_moves_tail.append(zero_move.clone())
        seq_moves: List[torch.Tensor] = [zero_move.clone() for _ in range(pad_count)] + padded_moves_tail

        # Stack into tensors matching training format expected by player:
        # [B, seq_len, 1, gin_dim]
        context_board_embeddings = torch.stack(seq_boards, dim=0).unsqueeze(0)  # [1, S, 1, D]
        context_move_embeddings = torch.stack(seq_moves, dim=0).unsqueeze(0)    # [1, S, 1, D]

        # Current board embedding, keep batch and "time" dims consistent with training API if needed
        current_board_embedding = cur.unsqueeze(0)  # [1, 1, gin]

        return context_board_embeddings, context_move_embeddings, current_board_embedding, legal_move_embeddings

    # --- Brain API ---
    def calculate_best_move(self, board: Board, restriction: str, value: int) -> str:
        # Check if we need to reset game history (new game detected)
        if (board.state == GameState.NOT_STARTED or 
            (self._last_board_state is not None and 
             self._last_board_state not in (GameState.IN_PROGRESS, GameState.NOT_STARTED) and
             board.state in (GameState.IN_PROGRESS, GameState.NOT_STARTED))):
            self._reset_game_history()
        
        self._last_board_state = board.state
        
        # Early exit if game over
        if board.state not in (GameState.IN_PROGRESS, GameState.NOT_STARTED):
            return Move.PASS

        # Current board embedding
        cur_emb = self._embed_board(board)
        if cur_emb is None:
            moves = list(board.get_valid_moves())
            return board.stringify_move(moves[0]) if moves else Move.PASS

        # Track opponent's move if they moved since our last move
        if self._last_board_after_our_move is not None:
            # Opponent moved: add their board state and inferred move to game history
            opp_move_emb = cur_emb - self._last_board_after_our_move
            self.game_board_embeddings.append(self._last_board_after_our_move.clone())
            self.game_move_embeddings.append(opp_move_emb.clone())

        # Legal moves for current position
        valid_moves = list(board.get_valid_moves())
        if not valid_moves:
            return Move.PASS

        after_embeds = self._batch_after_embeddings(board, valid_moves)  # List[[1, gin]]
        move_embeds = [ae - cur_emb for ae in after_embeds]              # difference mode
        legal_move_embeddings = torch.stack(move_embeds).unsqueeze(0)    # [1, num_moves, embedding_dim]
        legal_move_masks = torch.ones(legal_move_embeddings.size(1), dtype=torch.bool).unsqueeze(0)

        # Build training-style prompt with game history
        context_board_embeddings, context_move_embeddings, current_board_embedding, legal_move_embeddings = self._build_training_style_prompt(
            cur_emb, legal_move_embeddings, legal_move_masks
        )

        # Move tensors to LLM device
        llm_device = next(self.player.base_model.parameters()).device
        context_board_embeddings = context_board_embeddings.to(llm_device)
        context_move_embeddings = context_move_embeddings.to(llm_device)
        current_board_embedding = current_board_embedding.to(llm_device)
        legal_move_embeddings = legal_move_embeddings.to(llm_device)
        legal_move_masks = legal_move_masks.to(llm_device)

        # Generate move using LLM
        with torch.no_grad():
            best_idx, sim, pred_move, pred_state = self.player.generate_move(
                context_board_embeddings,
                context_move_embeddings,
                current_board_embedding,
                legal_move_embeddings,
                legal_move_masks,
            )

        # Choose best move and update game history
        best_i = int(best_idx)
        best_move = valid_moves[best_i] if 0 <= best_i < len(valid_moves) else valid_moves[0]
        best_move_str = board.stringify_move(best_move)

        # Update game history with our move
        chosen_after = after_embeds[best_i]
        chosen_move_emb = move_embeds[best_i]
        
        # Add our board state and move to game history
        self.game_board_embeddings.append(cur_emb.clone())
        self.game_move_embeddings.append(chosen_move_emb.clone())
        
        # Track board state after our move for opponent move inference
        self._last_board_after_our_move = chosen_after.clone()

        self._cache = best_move_str
        return best_move_str

    def empty_cache(self):
        """Override to also reset game history when cache is emptied"""
        super().empty_cache()
        self._reset_game_history()
