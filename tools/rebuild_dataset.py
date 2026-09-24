"""Rebuild the training set from the board.txt files, fixing the defects of the original.

The published dataset (Hive_dataset.rar) ships one JSON per position, but those JSONs are
redundant: x, edge_index and v are all recoverable from each game's board.txt. Rebuilding
from board.txt fixes, by construction, the three defects documented in
TRAINING_and_DATASET.md:

  1. the articulation feature was stale by one ply. Here it comes from the current engine
     at the position it describes.
  2. move_N.json counted plies skipping passes, so it was not "after N-1 plies". Here
     positions are indexed by the real ply, so passes cannot misalign anything.
  3. gen_dataset/graph_db_converter.py would emit inverted labels (its counter starts at
     -1.0, so for a White win the side to move is labelled as losing). Here the label is
     derived directly from the recorded outcome and the side to move.

It also records the move actually played at each position, which the original does not
store and which cannot be recovered later without replaying everything. Nothing in this
script trains on it; it is there for a future policy head.

Input layout (only board.txt is read):

    <src>/<collection>/game_<k>/board.txt

Output: one or more .npz shards per collection in the pre-collated layout PyG's
InMemoryDataset uses, plus a manifest. Loose files are avoided on purpose: a million small
files is painful on an HPC filesystem, and a list of a million Data objects does not fit
comfortably in RAM.

    python tools/rebuild_dataset.py --src data/raw --out data/shards
"""
import argparse
import fnmatch
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ai.board_graph import NUM_FEATURES, board_to_graph  # noqa: E402
from ai.move_index import MAX_DST, index_moves  # noqa: E402
from ai.oracle import Oracle  # noqa: E402
from engine.board import Board  # noqa: E402

# The hand-written heuristic, stored per position so training has a baseline to beat.
# It only counts queen neighbours, so it costs nothing to evaluate here.
_HEURISTIC = Oracle()

# nokamute-* are engine self-play games, so they belong with the bot games.
DEFAULT_COLLECTIONS = ("*tournament*", "*bots*", "nokamute*")
DECIDED = {"WhiteWins", "BlackWins", "Draw"}


def _label(outcome, white_to_move):
    """Training target in [0, 1]: the probability that the side to move wins.

    1.0 the side to move went on to win, 0.0 it lost, 0.5 a draw. This is already the
    (v + 1) / 2 that GraphDataset._process_data used to apply to the raw v in {-1, 0, 1},
    so the convention matches inference, where the network's sigmoid output is read as
    "probability the side to move wins".
    """
    if outcome == "Draw":
        return 0.5
    white_wins = outcome == "WhiteWins"
    return 1.0 if white_to_move == white_wins else 0.0


def rebuild_game(args):
    """Replay one game and emit every position. Returns None if the game is unusable."""
    collection, game_id, board_path = args
    try:
        raw = open(board_path).read()
    except OSError:
        return None
    # The 622 board.txt files of nokamute-6-test have the UHP "ok" token on a second
    # line, which otherwise ends up glued to the last move and makes the whole game
    # unparseable. Take the first non-empty line and ignore any protocol chatter.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ("skip", collection, game_id, "empty board.txt")
    gamestring = lines[0]
    parts = gamestring.split(";")
    if len(parts) < 3:
        return ("skip", collection, game_id, "malformed gamestring")
    outcome = parts[1]
    if outcome not in DECIDED:
        # Unfinished games carry no outcome, so there is nothing to learn from them.
        return ("skip", collection, game_id, f"undecided ({outcome})")
    moves = parts[3:]

    xs, edges, ys, plies, played, heur = [], [], [], [], [], []
    # Policy targets: every legal move expressed against the graph's nodes, plus which
    # one was actually played. Cheap to record now, impossible to recover later without
    # replaying everything again.
    mv_src, mv_bug, mv_dst, mv_count, mv_played = [], [], [], [], []
    try:
        board = Board(parts[0] or "Base+MLP")
        for ply in range(len(moves) + 1):
            built = board_to_graph(board)
            if built is not None:
                x, edge_index = built
                legal = list(board.get_valid_moves())
                _, src, bug, dst, _ = index_moves(board, legal)
                target = -1
                if ply < len(moves):
                    san = moves[ply].strip()
                    if san != "pass":
                        chosen = board._parse_move(san)
                        for j, candidate in enumerate(legal):
                            if candidate == chosen:
                                target = j
                                break
                xs.append(x)
                edges.append(edge_index)
                ys.append(_label(outcome, ply % 2 == 0))
                plies.append(ply)
                played.append(moves[ply] if ply < len(moves) else "")
                heur.append(_HEURISTIC.compute_heuristic(board))
                mv_src.append(src.astype(np.int8))
                mv_bug.append(bug.astype(np.int8))
                mv_dst.append(dst.astype(np.int8))
                mv_count.append(len(legal))
                mv_played.append(target)
            if ply < len(moves):
                board.play(moves[ply])
    except Exception as exc:  # a handful of games in the corpus do not replay
        return ("skip", collection, game_id, f"{type(exc).__name__}: {exc}"[:120])

    if not xs:
        return ("skip", collection, game_id, "no non-empty position")
    return ("ok", collection, game_id, xs, edges, ys, plies, played, heur,
            mv_src, mv_bug, mv_dst, mv_count, mv_played, len(moves), outcome)


class ShardWriter:
    """Accumulates graphs and flushes them to .npz shards of a bounded size."""

    def __init__(self, out_dir, collection, shard_size):
        self.out_dir = out_dir
        self.collection = collection
        self.shard_size = shard_size
        self.shards = []
        self._reset()

    def _reset(self):
        self.xs, self.edges, self.ys = [], [], []
        self.game_ids, self.plies, self.moves, self.heur = [], [], [], []
        self.mv_src, self.mv_bug, self.mv_dst = [], [], []
        self.mv_count, self.mv_played = [], []

    def add(self, game_id, xs, edges, ys, plies, played, heur,
            mv_src, mv_bug, mv_dst, mv_count, mv_played):
        self.xs.extend(xs)
        self.edges.extend(edges)
        self.ys.extend(ys)
        self.plies.extend(plies)
        self.moves.extend(played)
        self.heur.extend(heur)
        self.mv_src.extend(mv_src)
        self.mv_bug.extend(mv_bug)
        self.mv_dst.extend(mv_dst)
        self.mv_count.extend(mv_count)
        self.mv_played.extend(mv_played)
        self.game_ids.extend([game_id] * len(xs))
        if len(self.ys) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.ys:
            return
        index = len(self.shards)
        name = f"{self.collection}.{index:03d}"
        node_counts = np.fromiter((x.shape[0] for x in self.xs), dtype=np.int64, count=len(self.xs))
        edge_counts = np.fromiter((e.shape[1] for e in self.edges), dtype=np.int64, count=len(self.edges))
        node_ptr = np.zeros(len(self.xs) + 1, dtype=np.int64)
        edge_ptr = np.zeros(len(self.edges) + 1, dtype=np.int64)
        move_ptr = np.zeros(len(self.mv_count) + 1, dtype=np.int64)
        np.cumsum(node_counts, out=node_ptr[1:])
        np.cumsum(edge_counts, out=edge_ptr[1:])
        np.cumsum(np.asarray(self.mv_count, dtype=np.int64), out=move_ptr[1:])

        np.savez(
            os.path.join(self.out_dir, name + ".npz"),
            x=np.concatenate(self.xs, axis=0) if self.xs else np.zeros((0, NUM_FEATURES), np.float32),
            edge_index=np.concatenate(self.edges, axis=1) if self.edges else np.zeros((2, 0), np.int64),
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            y=np.asarray(self.ys, dtype=np.float32),
            game_id=np.asarray(self.game_ids, dtype=np.int32),
            ply=np.asarray(self.plies, dtype=np.int32),
            heuristic=np.asarray(self.heur, dtype=np.float32),
            # Policy side. Node indices fit in an int8 because a hive never exceeds 28
            # pieces; -1 means "in hand" for src and padding for dst.
            move_ptr=move_ptr,
            move_src=np.concatenate(self.mv_src) if self.mv_src else np.zeros(0, np.int8),
            move_bug=np.concatenate(self.mv_bug) if self.mv_bug else np.zeros(0, np.int8),
            move_dst=(np.concatenate(self.mv_dst) if self.mv_dst
                      else np.zeros((0, MAX_DST), np.int8)),
            move_played=np.asarray(self.mv_played, dtype=np.int16),
        )
        # One line per graph, same order as y: the move played at that position, empty on
        # the terminal one. Kept out of the npz because fixed-width unicode arrays waste
        # a lot of space for short move strings.
        with open(os.path.join(self.out_dir, name + ".moves.txt"), "w") as handle:
            handle.write("\n".join(self.moves))

        self.shards.append({"name": name, "graphs": len(self.ys),
                            "nodes": int(node_ptr[-1]), "edges": int(edge_ptr[-1]),
                            "moves": int(move_ptr[-1])})
        self._reset()


def find_games(src, patterns):
    """Yield (collection, game_id, board_path) for every game in the matching folders."""
    for collection in sorted(os.listdir(src)):
        path = os.path.join(src, collection)
        if not os.path.isdir(path) or collection == "BAK":
            continue
        if not any(fnmatch.fnmatch(collection, p) for p in patterns):
            continue
        games = [d for d in os.listdir(path) if d.startswith("game_")]
        games.sort(key=lambda d: int(d.split("_")[1]))
        for game in games:
            board_path = os.path.join(path, game, "board.txt")
            if os.path.exists(board_path):
                yield collection, int(game.split("_")[1]), board_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="directory holding the collection folders")
    parser.add_argument("--out", required=True, help="where to write the shards")
    parser.add_argument("--collections", nargs="*", default=list(DEFAULT_COLLECTIONS),
                        help="glob patterns over collection folder names "
                             f"(default: {' '.join(DEFAULT_COLLECTIONS)})")
    parser.add_argument("--shard-size", type=int, default=100_000,
                        help="graphs per shard (default 100000)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="parallel replay workers")
    parser.add_argument("--limit", type=int, default=0, help="stop after N games (for testing)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    games = list(find_games(args.src, args.collections))
    if args.limit:
        games = games[:args.limit]
    if not games:
        raise SystemExit(f"no games found under {args.src} matching {args.collections}")

    collections = sorted({c for c, _, _ in games})
    print(f"{len(games)} games in {len(collections)} collections: {', '.join(collections)}")

    writers = {c: ShardWriter(args.out, c, args.shard_size) for c in collections}
    skipped, kept, positions = [], 0, 0
    start = time.perf_counter()

    # Games are independent, so replay them in parallel and collect in submission order
    # to keep the output deterministic.
    with Pool(max(1, args.workers)) as pool:
        for i, result in enumerate(pool.imap(rebuild_game, games, chunksize=16)):
            if result is None or result[0] == "skip":
                if result is not None:
                    skipped.append({"collection": result[1], "game": result[2], "reason": result[3]})
                continue
            (_, collection, game_id, xs, edges, ys, plies, played, heur,
             mv_src, mv_bug, mv_dst, mv_count, mv_played, n_plies, outcome) = result
            writers[collection].add(game_id, xs, edges, ys, plies, played, heur,
                                    mv_src, mv_bug, mv_dst, mv_count, mv_played)
            kept += 1
            positions += len(ys)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.perf_counter() - start)
                print(f"  {i + 1}/{len(games)} games  {positions} positions  {rate:.0f} games/s",
                      flush=True)

    manifest = {
        "source": os.path.abspath(args.src),
        "collections": args.collections,
        "games_found": len(games),
        "games_kept": kept,
        "games_skipped": len(skipped),
        "positions": positions,
        "num_features": NUM_FEATURES,
        "label": "y in [0,1] = probability the side to move wins (0.5 = draw)",
        "heuristic": "ai.oracle.Oracle.compute_heuristic, same convention as y; the "
                     "baseline the network has to beat",
        "shards": {},
        "skipped": skipped[:200],
    }
    for collection, writer in writers.items():
        writer.flush()
        manifest["shards"][collection] = writer.shards

    with open(os.path.join(args.out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    elapsed = time.perf_counter() - start
    print(f"\nkept {kept}/{len(games)} games, {positions} positions, "
          f"{len(skipped)} skipped, in {elapsed:.1f}s")
    for collection, writer in writers.items():
        total = sum(s["graphs"] for s in writer.shards)
        print(f"  {collection:<32} {total:>8} graphs in {len(writer.shards)} shard(s)")
    if skipped:
        print("\nfirst skipped games:")
        for s in skipped[:5]:
            print(f"  {s['collection']}/game_{s['game']}: {s['reason']}")


if __name__ == "__main__":
    main()
