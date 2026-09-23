"""Reproducible benchmark for the MCTS hot path.

Runs without torch: the GNN oracle is replaced by the pure-numpy heuristic Oracle,
so the numbers measure the game engine and the search, which is where the time goes.

    python bench/bench_engine.py            # full run
    python bench/bench_engine.py --quick    # shorter, for tight iteration
"""
import argparse
import ast
import os
import random
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

SEED = 20260923


def _load_positions():
    """Pull the gamestrings out of src/test_mcts.py without importing it (it needs torch).

    Only the testcases with win=False are kept: a position with a mate in one ends the
    rollout at the first terminal child and measures nothing.
    """
    source = open(os.path.join(ROOT, "src", "test_mcts.py")).read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "testcases" for t in node.targets
        ):
            cases = ast.literal_eval(node.value)
            return [c["start"] for c in cases if not c["win"]]
    raise RuntimeError("testcases not found in src/test_mcts.py")


POSITIONS = _load_positions()


def _timer():
    return time.perf_counter()


def bench_board_construction(reps):
    from engine.board import Board
    t = _timer()
    for _ in range(reps):
        Board()
    return (_timer() - t) / reps


def bench_play_undo(board, moves, reps):
    t = _timer()
    for _ in range(reps):
        for m in moves:
            board.safe_play(m)
            board.undo()
    return (_timer() - t) / (reps * len(moves))


def bench_valid_moves_cold(gamestring, reps):
    """Cost of generating the legal moves with an empty snapshot cache."""
    from engine.board import Board
    board = Board(gamestring)
    t = _timer()
    for _ in range(reps):
        board._snapshots.clear()
        board.get_valid_moves()
    return (_timer() - t) / reps


def bench_mcts(gamestring, rollouts):
    from ai.brains import MCTS
    from ai.oracle import Oracle
    from engine.board import Board
    random.seed(SEED)
    board = Board(gamestring)
    mcts = MCTS(oracle=Oracle(), exploration_weight=5, num_rollouts=rollouts)
    t = _timer()
    mcts.run_simulation_from(board)
    elapsed = _timer() - t
    return rollouts / elapsed, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rollouts", type=int, default=0)
    args = parser.parse_args()

    board_reps = 1 if args.quick else 3
    play_reps = 2000 if args.quick else 10000
    cold_reps = 200 if args.quick else 1000
    rollouts = args.rollouts or (200 if args.quick else 1000)

    from engine.board import Board

    print(f"seed={SEED}")
    print("=" * 62)

    dt = bench_board_construction(board_reps)
    print(f"{'Board() construction':<40} {dt * 1e3:10.2f} ms")

    for i, gs in enumerate(POSITIONS):
        board = Board(gs)
        moves = sorted(board.get_valid_moves(), key=str)[:5]
        if not moves:
            continue
        dt = bench_play_undo(board, moves, play_reps // 5)
        print(f"{'safe_play+undo  pos' + str(i):<40} {dt * 1e6:10.2f} us")

    for i, gs in enumerate(POSITIONS):
        dt = bench_valid_moves_cold(gs, cold_reps)
        print(f"{'get_valid_moves (cold)  pos' + str(i):<40} {dt * 1e6:10.2f} us")

    print("-" * 62)
    total = 0.0
    for i, gs in enumerate(POSITIONS):
        rate, elapsed = bench_mcts(gs, rollouts)
        total += rate
        print(f"{'MCTS rollouts/s  pos' + str(i):<40} {rate:10.1f} /s   ({elapsed:.2f}s)")
    print(f"{'MCTS rollouts/s  TOTAL':<40} {total:10.1f} /s")
    print("=" * 62)


if __name__ == "__main__":
    main()
