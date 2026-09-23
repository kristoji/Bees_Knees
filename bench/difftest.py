"""Deterministic fingerprint of the engine's observable behaviour.

Prints one digest covering, for a set of deterministic playouts: every legal move
set encountered, the board state and the turn number at every ply. Two revisions
that agree on the digest generate the same moves from the same positions.

The zobrist key is deliberately NOT part of the digest: the random tables it is
built from are an implementation detail, and changing them changes every key
without changing the game. It is still checked, as an invariant — every move must
undo back to the exact key it started from.

    BENCH_SRC=/path/to/src python bench/difftest.py

Compare two revisions with bench/diffcheck.py.
"""
import ast
import hashlib
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.environ.get("BENCH_SRC") or os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

SEED = 4242
PLIES = 120


def _load_positions():
    source = open(os.path.join(SRC, "test_mcts.py")).read()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "testcases" for t in node.targets
        ):
            return [c["start"] for c in ast.literal_eval(node.value)]
    raise RuntimeError("testcases not found")


def fingerprint():
    from engine.board import Board
    from engine.enums import GameState

    h = hashlib.sha256()
    starts = ["", "Base+MLP"] + _load_positions()

    for start_index, start in enumerate(starts):
        rng = random.Random(SEED + start_index)
        board = Board(start)
        for ply in range(PLIES):
            moves = board.get_valid_moves()
            # Move sets are unordered, so sort the rendered strings before hashing.
            rendered = sorted(board.stringify_move(m) for m in moves)
            h.update(f"{start_index}|{ply}|{board.state}|{board.turn}|".encode())
            h.update("\x1f".join(rendered).encode())

            if not moves:
                # Only a pass is available.
                if board.state is not GameState.IN_PROGRESS and board.state is not GameState.NOT_STARTED:
                    break
                board.safe_play(None)
                continue

            ordered = sorted(moves, key=lambda m: board.stringify_move(m))
            chosen = ordered[rng.randrange(len(ordered))]

            # Every legal move must be undoable back to the exact same key.
            before = board.zobrist_key
            board.safe_play(chosen)
            after_state = board.state
            board.undo()
            assert board.zobrist_key == before, f"undo broke the key at {start_index}/{ply}"

            board.safe_play(chosen)
            assert board.state == after_state
            if board.state is not GameState.IN_PROGRESS:
                h.update(f"|end:{board.state}".encode())
                break
    return h.hexdigest()


if __name__ == "__main__":
    print(fingerprint())
