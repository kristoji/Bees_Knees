"""Play the trained network against the hand-written heuristic, same search, same budget.

Validation loss says how well the network predicts the outcome of positions drawn from
the training distribution. It does not say whether the tree plays better, which is the
only thing that matters here. This pits MCTS_BATCH driven by the GNN against MCTS_BATCH
driven by ai.oracle.Oracle, at an identical rollout count, alternating colours.

The heuristic is wrapped in an adapter that exposes the two methods MCTS_BATCH calls, so
both sides run the *same* searcher and only the value function differs.

    python tools/duel_oracles.py --weights models/best.pt --summary models/summary.json \
        --games 20 --rollouts 100
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ai.mcts_batch import MCTS_BATCH  # noqa: E402
from ai.oracle import Oracle  # noqa: E402
from ai.oracleGNN import OracleGNN  # noqa: E402
from engine.board import Board  # noqa: E402
from engine.enums import GameState, PlayerColor  # noqa: E402


class _Value:
    """Stands in for a PyG Data: carries one precomputed heuristic value."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


class HeuristicOracle:
    """Adapts ai.oracle.Oracle to the interface MCTS_BATCH expects."""

    def __init__(self):
        self._oracle = Oracle()

    def _data_from_board(self, board):
        return _Value(self._oracle.compute_heuristic(board))

    def predict_values_batch_from_data(self, data_list, use_sigmoid=True):
        return [d.v for d in data_list]


def load_gnn(weights, summary_path, device):
    cfg = json.load(open(summary_path))["args"]
    oracle = OracleGNN(
        device=device, hidden_dim=cfg["hidden_dim"], conv_type=cfg["conv_type"],
        num_layers=cfg["num_layers"], gat_heads=cfg["gat_heads"],
        conv_dropout=cfg["dropout"], mlp_dropout=cfg["dropout"],
        final_dropout=cfg["dropout"], use_layer_norm=True, use_residual=False,
        pooling=cfg["pooling"], mlp_layers=2, final_mlp_layers=2,
    )
    oracle.load(weights)
    return oracle, cfg


def play_game(oracle_white, oracle_black, rollouts, max_plies, exploration, seed):
    random.seed(seed)
    board = Board("Base+MLP")
    searchers = {
        PlayerColor.WHITE: MCTS_BATCH(oracle=oracle_white, exploration_weight=exploration,
                                      num_rollouts=rollouts, batch_size=32),
        PlayerColor.BLACK: MCTS_BATCH(oracle=oracle_black, exploration_weight=exploration,
                                      num_rollouts=rollouts, batch_size=32),
    }
    for _ in range(max_plies):
        move = searchers[board.current_player_color].calculate_best_move(
            board, restriction="depth", value=rollouts)
        board.play(move)
        if board.state is not GameState.IN_PROGRESS:
            return board.state, board.turn
    return None, board.turn   # hit the ply cap


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=150)
    parser.add_argument("--exploration", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--out", default=None, help="write the result as JSON here")
    args = parser.parse_args()

    gnn, cfg = load_gnn(args.weights, args.summary, args.device)
    heuristic = HeuristicOracle()
    print(f"GNN: {cfg['conv_type']} x{cfg['num_layers']}, hidden {cfg['hidden_dim']}, "
          f"from {args.weights}")
    print(f"{args.games} games, {args.rollouts} rollouts per move, colours alternate\n")

    wins = losses = draws = capped = 0
    start = time.perf_counter()
    for game in range(args.games):
        gnn_is_white = game % 2 == 0
        white = gnn if gnn_is_white else heuristic
        black = heuristic if gnn_is_white else gnn
        state, plies = play_game(white, black, args.rollouts, args.max_plies,
                                 args.exploration, args.seed + game)
        if state is None:
            capped += 1
            result = "cap"
        elif state is GameState.DRAW:
            draws += 1
            result = "draw"
        else:
            gnn_won = ((state is GameState.WHITE_WINS) == gnn_is_white)
            wins += gnn_won
            losses += not gnn_won
            result = "GNN wins" if gnn_won else "heuristic wins"
        print(f"  game {game + 1:3d}  GNN as {'White' if gnn_is_white else 'Black'}  "
              f"{plies:3d} plies  {result}", flush=True)

    decided = wins + losses
    elapsed = time.perf_counter() - start
    # Games that hit the ply cap are counted as draws: neither side converted an
    # advantage, so scoring them either way would overstate the result.
    score = (wins + 0.5 * (draws + capped)) / max(1, args.games)
    print(f"\nGNN {wins} - {losses} heuristic, {draws} draws, {capped} hit the ply cap")
    print(f"score {score * 100:.1f}%  (decided games: {wins}/{decided})" if decided
          else f"score {score * 100:.1f}%  (no decided games)")
    print(f"{elapsed:.0f}s, {elapsed / max(1, args.games):.1f}s per game")

    if args.out:
        json.dump({"wins": wins, "losses": losses, "draws": draws, "capped": capped,
                   "score": score, "games": args.games, "rollouts": args.rollouts,
                   "weights": args.weights, "seconds": elapsed},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
