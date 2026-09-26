"""Generate nokamute self-play games as board.txt files the rebuild can consume.

We lose to nokamute, so we are below its ceiling rather than against it. That makes
more games from it the cheapest useful data available: the corpus holds 2621 nokamute
games out of 15325, and one more costs a few seconds.

nokamute plays itself with `nokamute play --depth=N ai ai` and prints the whole move
list. Its games at a fixed depth are NOT identical run to run, which is the first thing
this checks for, because a deterministic generator would write the same game N times.

Output matches the corpus layout so tools/rebuild_dataset.py reads it with no changes:

    <out>/nokamute-dN/game_<k>/board.txt

    python tools/gen_nokamute.py --engine path/to/nokamute --out data/raw \
        --games 5000 --depth 5 --workers 8
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from engine.board import Board  # noqa: E402
from engine.enums import GameState  # noqa: E402

DECIDED = (GameState.WHITE_WINS, GameState.BLACK_WINS, GameState.DRAW)


def one_game(job):
    """Run one nokamute self-play game and return its gamestring, or a reason to skip."""
    engine, strength, timeout, index = job
    # --num-threads=1 matters: nokamute otherwise takes every core, and several
    # generators then fight each other instead of running in parallel.
    cmd = [engine, "--num-threads=1", "play", *strength, "ai", "ai"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return index, None, "timeout"
    log = None
    for line in proc.stdout.splitlines():
        if line.startswith("Game log:"):
            log = line[len("Game log:"):].strip()
    if not log:
        return index, None, "no game log"

    moves = [m.strip() for m in log.split(";") if m.strip()]
    board = Board("Base+MLP")
    try:
        for move in moves:
            board.play(move)
    except Exception as exc:
        # Our engine agrees with nokamute's move generator on its own testsuite, so a
        # rejection here is worth counting rather than ignoring.
        return index, None, f"replay failed at ply {board.turn}: {type(exc).__name__}"
    if board.state not in DECIDED:
        return index, None, f"undecided ({board.state})"
    return index, str(board), None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--out", required=True, help="the directory of collections")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--depth", type=int, default=0,
                        help="fixed search depth; cheap but a weaker teacher")
    parser.add_argument("--move-time", default=None,
                        help="seconds per move instead of a depth, as nokamute wants "
                             "it: whole seconds with a unit, e.g. 1s. Stronger and "
                             "much slower than --depth")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    if args.move_time:
        strength = [f"--timeout={args.move_time}"]
        label = f"t{args.move_time}"
    elif args.depth:
        strength = [f"--depth={args.depth}"]
        label = f"d{args.depth}"
    else:
        raise SystemExit("give either --depth or --move-time")
    collection = args.collection or f"nokamute-{label}"
    root = os.path.join(args.out, collection)
    os.makedirs(root, exist_ok=True)

    # A deterministic engine would write the same game every time; check before
    # spending hours on it.
    probe = [one_game((args.engine, strength, args.timeout, i)) for i in range(3)]
    logs = {g for _, g, _ in probe if g}
    if len(logs) < 2:
        raise SystemExit(f"nokamute produced {len(logs)} distinct game(s) in 3 tries "
                         f"at {label}: it looks deterministic, and generating more "
                         f"would just duplicate one game")
    print(f"{collection}: {len(logs)}/3 distinct games in the probe, proceeding")

    jobs = [(args.engine, strength, args.timeout, i)
            for i in range(args.start_game, args.start_game + args.games)]
    kept, skipped, reasons = 0, 0, {}
    start = time.perf_counter()
    with Pool(max(1, args.workers)) as pool:
        for n, (index, gamestring, problem) in enumerate(pool.imap_unordered(one_game, jobs, 8)):
            if gamestring is None:
                skipped += 1
                reasons[problem.split(":")[0]] = reasons.get(problem.split(":")[0], 0) + 1
                continue
            game_dir = os.path.join(root, f"game_{index}")
            os.makedirs(game_dir, exist_ok=True)
            with open(os.path.join(game_dir, "board.txt"), "w") as handle:
                handle.write(gamestring)
            kept += 1
            if (n + 1) % 250 == 0:
                rate = (n + 1) / (time.perf_counter() - start)
                print(f"  {n + 1}/{len(jobs)}  {rate:.1f} games/s", flush=True)

    elapsed = time.perf_counter() - start
    print(f"\nkept {kept}, skipped {skipped} in {elapsed:.0f}s "
          f"({kept / max(1e-9, elapsed):.1f} games/s)")
    if reasons:
        print(f"skip reasons: {reasons}")
    print(f"written to {root}")


if __name__ == "__main__":
    main()
