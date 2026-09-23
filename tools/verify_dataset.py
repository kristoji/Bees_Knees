"""Check a rebuilt dataset: invariants, statistics, and a diff against the original JSONs.

    python tools/verify_dataset.py --shards data/shards
    python tools/verify_dataset.py --shards data/shards --against data/raw

With --against it replays every game and compares the rebuilt graphs to the JSONs that
shipped with the dataset, using the original's own indexing (move_N is the position before
the N-th non-pass move). The expected outcome is:

  - columns 0..11 and the edge list identical everywhere. Those were already correct.
  - column 12 (articulation) differing on many positions, and matching the engine rather
    than the JSON. That is the fix, not a regression.
  - labels identical everywhere.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ai.board_graph import NUM_FEATURES, board_to_graph  # noqa: E402
from engine.board import Board  # noqa: E402

DECIDED = {"WhiteWins", "BlackWins", "Draw"}


def load_shards(shard_dir, collections=None):
    for path in sorted(glob.glob(os.path.join(shard_dir, "*.npz"))):
        name = os.path.basename(path)[:-4]
        collection = name.rsplit(".", 1)[0]
        if collections and collection not in collections:
            continue
        yield collection, path, np.load(path)


def check_invariants(shard_dir):
    manifest_path = os.path.join(shard_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    total_graphs = total_nodes = 0
    label_counts = {}
    node_sizes = []
    problems = []

    for collection, path, data in load_shards(shard_dir):
        x, edge_index = data["x"], data["edge_index"]
        node_ptr, edge_ptr = data["node_ptr"], data["edge_ptr"]
        y, game_id, ply = data["y"], data["game_id"], data["ply"]
        n = len(y)

        if x.shape[1] != NUM_FEATURES:
            problems.append(f"{path}: {x.shape[1]} features, expected {NUM_FEATURES}")
        if node_ptr[-1] != x.shape[0]:
            problems.append(f"{path}: node_ptr ends at {node_ptr[-1]}, x has {x.shape[0]} rows")
        if edge_ptr[-1] != edge_index.shape[1]:
            problems.append(f"{path}: edge_ptr ends at {edge_ptr[-1]}, "
                            f"edge_index has {edge_index.shape[1]}")
        if len(node_ptr) != n + 1 or len(edge_ptr) != n + 1:
            problems.append(f"{path}: pointer arrays do not match {n} graphs")
        if not np.isfinite(x).all():
            problems.append(f"{path}: non-finite values in x")
        if ((y < 0) | (y > 1)).any():
            problems.append(f"{path}: labels outside [0, 1]")
        if np.any(np.diff(node_ptr) <= 0):
            problems.append(f"{path}: a graph has no nodes")

        # Edge endpoints must stay inside their own graph.
        for i in range(min(n, 500)):
            lo, hi = node_ptr[i], node_ptr[i + 1]
            e = edge_index[:, edge_ptr[i]:edge_ptr[i + 1]]
            if e.size and (e.min() < 0 or e.max() >= hi - lo):
                problems.append(f"{path}: graph {i} has an edge outside its node range")
                break

        total_graphs += n
        total_nodes += x.shape[0]
        node_sizes.append(np.diff(node_ptr))
        for value, count in zip(*np.unique(y, return_counts=True)):
            label_counts[float(value)] = label_counts.get(float(value), 0) + int(count)
        moves_path = path[:-4] + ".moves.txt"
        if os.path.exists(moves_path):
            with open(moves_path) as handle:
                n_moves = sum(1 for _ in handle) + 1
            if n_moves != n:
                problems.append(f"{moves_path}: {n_moves} lines for {n} graphs")

    sizes = np.concatenate(node_sizes) if node_sizes else np.zeros(0)
    print(f"graphs      : {total_graphs}")
    print(f"nodes       : {total_nodes}  (mean {sizes.mean():.1f}, max {int(sizes.max())})"
          if total_graphs else "nodes       : 0")
    print("labels      : " + ", ".join(
        f"{k:g} -> {v} ({v / max(1, total_graphs) * 100:.1f}%)"
        for k, v in sorted(label_counts.items())))
    if manifest:
        print(f"games kept  : {manifest.get('games_kept')} / {manifest.get('games_found')} "
              f"({manifest.get('games_skipped')} skipped)")
    print(f"problems    : {len(problems)}")
    for p in problems[:10]:
        print("  " + p)
    return not problems


def _flatten(row):
    out = []
    for element in row:
        out.extend(element) if isinstance(element, list) else out.append(element)
    return out


def compare_against(src, collections):
    """Replay every game and diff the rebuilt graph against the shipped JSON."""
    identical = art_only = other = label_mismatch = missing = 0
    art_agree_engine = 0
    games = 0

    for collection in sorted(os.listdir(src)):
        path = os.path.join(src, collection)
        if not os.path.isdir(path) or collection == "BAK":
            continue
        if collections and collection not in collections:
            continue
        for game in sorted(os.listdir(path), key=lambda d: int(d.split("_")[1])
                           if d.startswith("game_") else -1):
            board_path = os.path.join(path, game, "board.txt")
            if not game.startswith("game_") or not os.path.exists(board_path):
                continue
            parts = open(board_path).read().strip().split(";")
            if len(parts) < 3 or parts[1] not in DECIDED:
                continue
            games += 1
            moves = parts[3:]
            board = Board(parts[0] or "Base+MLP")
            json_index = 1
            for ply in range(len(moves) + 1):
                json_path = os.path.join(path, game, f"move_{json_index}.json")
                # The original only saved a position when the move about to be played was
                # not a pass (plus one extra for the terminal position).
                saved_here = ply == len(moves) or moves[ply].strip() != "pass"
                built = board_to_graph(board)
                if saved_here and built is not None:
                    if not os.path.exists(json_path):
                        missing += 1
                    else:
                        blob = json.load(open(json_path))
                        want = (np.array([_flatten(r) for r in blob["x"]], dtype=np.float32)
                                if blob["x"] else None)
                        x, edge_index = built
                        want_edges = (sorted(zip(*blob["edge_index"]))
                                      if blob["edge_index"][0] else [])
                        got_edges = sorted(map(tuple, edge_index.T.tolist()))
                        if want is None or want.shape != x.shape or got_edges != want_edges:
                            other += 1
                        elif np.array_equal(x, want):
                            identical += 1
                        elif np.array_equal(x[:, :12], want[:, :12]):
                            art_only += 1
                            art_agree_engine += 1
                        else:
                            other += 1
                        expected = (0.5 if parts[1] == "Draw"
                                    else 1.0 if (ply % 2 == 0) == (parts[1] == "WhiteWins")
                                    else 0.0)
                        if abs((blob["v"] + 1) / 2 - expected) > 1e-6:
                            label_mismatch += 1
                if saved_here:
                    json_index += 1
                if ply < len(moves):
                    board.play(moves[ply])

    total = identical + art_only + other
    print(f"\ncompared {total} graphs from {games} games against the original JSONs")
    print(f"  identical                          : {identical}")
    print(f"  differ only in column 12 (the fix) : {art_only}")
    print(f"  differ elsewhere                   : {other}")
    print(f"  JSON missing where expected        : {missing}")
    print(f"  label disagreements                : {label_mismatch}")
    print("\ncolumns 0..11 and the edges must match everywhere; column 12 is expected to "
          "differ,\nbecause the original computed it one ply late.")
    return other == 0 and label_mismatch == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--against", help="directory of the original collection folders")
    parser.add_argument("--collections", nargs="*")
    args = parser.parse_args()

    ok = check_invariants(args.shards)
    if args.against:
        ok = compare_against(args.against, args.collections) and ok
    print("\nOK" if ok else "\nFAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
