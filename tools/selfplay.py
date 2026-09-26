"""Generate self-play games and write them as training shards.

Supervised training has a ceiling: the policy target is "the move that player played",
so imitating a corpus takes you to the level of the players in it and no further. Our
corpus contains nokamute games, and nokamute beats us, which is exactly the shape of
that ceiling.

Self-play breaks it because the target improves with the player. The policy target here
is the MCTS visit distribution, not a single move: the search with a few hundred
rollouts plays better than the raw policy head, and training the head toward the
search's own preferences is what pushes the network past whatever produced it.

Two things must be stochastic or every game comes out identical, which is what the
search did before: Dirichlet noise on the root priors, and sampling the played move
from the visit counts for the opening plies.

    python tools/selfplay.py --weights models/agent/best.pt \
        --summary models/agent/summary.json --games 50 --rollouts 400 --out data/selfplay

Shards match the layout tools/rebuild_dataset.py writes, with one addition: move_pi, the
visit distribution over each position's legal moves.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ai.board_graph import NUM_FEATURES, board_to_graph  # noqa: E402
from ai.mcts_batch import MCTS_BATCH  # noqa: E402
from ai.move_index import MAX_DST, index_moves  # noqa: E402
from ai.oracle import Oracle  # noqa: E402
from engine.board import Board  # noqa: E402
from engine.enums import GameState, PlayerColor  # noqa: E402

_HEURISTIC = Oracle()
LIVE = (GameState.IN_PROGRESS, GameState.NOT_STARTED)


def load_agent(weights, summary_path, device):
    from ai.oracleGNN import OracleGNN
    import torch

    cfg = json.load(open(summary_path))["args"]
    kwargs = dict(hidden_dim=cfg["hidden_dim"], conv_type=cfg["conv_type"],
                  num_layers=cfg["num_layers"], gat_heads=cfg["gat_heads"],
                  conv_dropout=cfg["dropout"], mlp_dropout=cfg["dropout"],
                  final_dropout=cfg["dropout"], use_layer_norm=True,
                  use_residual=cfg.get("residual", False), pooling=cfg["pooling"],
                  mlp_layers=2, final_mlp_layers=2)
    oracle = OracleGNN(device=device, **kwargs)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not any(k.startswith("policy.") for k in state):
        raise SystemExit("self-play needs a checkpoint with a policy head")
    oracle.load_agent(weights, **kwargs)
    return oracle, cfg


def play_one(oracle, args, seed):
    """Play a game against itself, returning one record per position."""
    np.random.seed(seed)
    board = Board("Base+MLP")
    searcher = MCTS_BATCH(oracle=oracle, exploration_weight=args.exploration,
                          num_rollouts=args.rollouts, batch_size=args.batch_size,
                          dirichlet_eps=args.dirichlet_eps,
                          dirichlet_alpha=args.dirichlet_alpha,
                          temperature_plies=args.temperature_plies)
    positions = []

    while board.turn < args.max_plies and board.state in LIVE:
        built = board_to_graph(board)
        legal = list(board.get_valid_moves())
        if not legal:
            board.safe_play(None)          # only a pass is available
            continue

        # Run the search and read the root's visit counts BEFORE choosing, because
        # action_selection() descends init_node into the chosen child and
        # get_moves_probs() would then describe the opponent's replies instead.
        searcher.time_limit = float("inf")
        searcher.num_rollouts = args.rollouts
        searcher.run_simulation_from(board)
        visits = searcher.get_moves_probs()
        move_str = searcher.action_selection(training=False)

        if built is not None:
            x, edge_index = built
            _, src, bug, dst, _ = index_moves(board, legal)
            pi = np.array([visits.get(m, 0.0) for m in legal], dtype=np.float32)
            total = pi.sum()
            if total <= 0:
                # Every legal move should carry some visit mass. Zero means the root
                # we read is not the root we searched, which is silent data loss.
                raise RuntimeError(
                    f"visit distribution does not cover the legal moves at ply "
                    f"{board.turn}: {len(visits)} visited, {len(legal)} legal")
            pi /= total
            positions.append({
                "x": x, "edge_index": edge_index, "ply": board.turn,
                "src": src.astype(np.int8), "bug": bug.astype(np.int8),
                "dst": dst.astype(np.int8), "pi": pi,
                "white_to_move": board.current_player_color is PlayerColor.WHITE,
                "heuristic": _HEURISTIC.compute_heuristic(board),
            })
        board.play(move_str)

    outcome = board.state
    if outcome not in (GameState.WHITE_WINS, GameState.BLACK_WINS, GameState.DRAW):
        return None, "hit the ply cap", board.turn   # no outcome, so no label
    return positions, outcome, board.turn


def label(outcome, white_to_move):
    if outcome is GameState.DRAW:
        return 0.5
    return 1.0 if (white_to_move == (outcome is GameState.WHITE_WINS)) else 0.0


def write_shard(out_dir, name, games):
    """Flatten the games into the same pre-collated layout as the supervised shards."""
    xs, edges, ys, plies, heur, gids = [], [], [], [], [], []
    srcs, bugs, dsts, pis, counts = [], [], [], [], []
    for game_id, (positions, outcome) in enumerate(games):
        for p in positions:
            xs.append(p["x"])
            edges.append(p["edge_index"])
            ys.append(label(outcome, p["white_to_move"]))
            plies.append(p["ply"])
            heur.append(p["heuristic"])
            gids.append(game_id)
            srcs.append(p["src"])
            bugs.append(p["bug"])
            dsts.append(p["dst"])
            pis.append(p["pi"])
            counts.append(len(p["pi"]))
    if not xs:
        return None

    node_ptr = np.zeros(len(xs) + 1, dtype=np.int64)
    edge_ptr = np.zeros(len(xs) + 1, dtype=np.int64)
    move_ptr = np.zeros(len(xs) + 1, dtype=np.int64)
    np.cumsum([a.shape[0] for a in xs], out=node_ptr[1:])
    np.cumsum([e.shape[1] for e in edges], out=edge_ptr[1:])
    np.cumsum(counts, out=move_ptr[1:])

    path = os.path.join(out_dir, name + ".npz")
    np.savez(
        path,
        x=np.concatenate(xs), edge_index=np.concatenate(edges, axis=1),
        node_ptr=node_ptr, edge_ptr=edge_ptr, move_ptr=move_ptr,
        y=np.asarray(ys, dtype=np.float32),
        game_id=np.asarray(gids, dtype=np.int32),
        ply=np.asarray(plies, dtype=np.int32),
        heuristic=np.asarray(heur, dtype=np.float32),
        move_src=np.concatenate(srcs), move_bug=np.concatenate(bugs),
        move_dst=np.concatenate(dsts),
        # The visit distribution replaces the single played move as the policy target.
        move_pi=np.concatenate(pis),
        move_played=np.full(len(xs), -1, dtype=np.int16),
    )
    return {"name": name, "graphs": len(xs), "nodes": int(node_ptr[-1]),
            "edges": int(edge_ptr[-1]), "moves": int(move_ptr[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exploration", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--temperature-plies", type=int, default=20,
                        help="plies sampled from the visit counts before going greedy")
    parser.add_argument("--keep-capped", action="store_true",
                        help="keep games that hit the ply cap, labelled as draws. Off "
                             "by default: the network's features say nothing about how "
                             "far along a game is, so a cap-induced draw is label noise")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--tag", default="selfplay")
    args = parser.parse_args()

    if args.dirichlet_eps <= 0 and args.temperature_plies <= 0:
        raise SystemExit("with no noise and no temperature the search is deterministic "
                         "and every game would be identical")

    oracle, cfg = load_agent(args.weights, args.summary, args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"agent   : {cfg['conv_type']} x{cfg['num_layers']}, hidden {cfg['hidden_dim']}")
    print(f"settings: {args.rollouts} rollouts, dirichlet {args.dirichlet_eps} "
          f"alpha {args.dirichlet_alpha}, temperature for {args.temperature_plies} plies")

    kept, dropped, start = [], 0, time.perf_counter()
    outcomes = {}
    for i in range(args.start_game, args.start_game + args.games):
        positions, outcome, plies = play_one(oracle, args, args.seed + i)
        if positions is None:
            if args.keep_capped:
                positions, outcome = [], GameState.DRAW
            dropped += 1
            print(f"  game {i:4d}  {plies:3d} plies  {outcome}", flush=True)
            continue
        kept.append((positions, outcome))
        outcomes[str(outcome)] = outcomes.get(str(outcome), 0) + 1
        print(f"  game {i:4d}  {plies:3d} plies  {outcome}  "
              f"{len(positions)} positions", flush=True)

    name = f"{args.tag}.{args.start_game:05d}"
    shard = write_shard(args.out, name, kept)
    elapsed = time.perf_counter() - start
    manifest = {"games": len(kept), "dropped_at_cap": dropped, "outcomes": outcomes,
                "shard": shard, "seconds": elapsed, "args": vars(args),
                "source_weights": args.weights, "num_features": NUM_FEATURES}
    with open(os.path.join(args.out, name + ".manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\nkept {len(kept)} games ({dropped} hit the cap and were dropped), "
          f"{shard['graphs'] if shard else 0} positions in {elapsed:.0f}s")
    print(f"outcomes: {outcomes}")
    if kept:
        print(f"{elapsed / len(kept):.1f}s per game")


if __name__ == "__main__":
    main()
