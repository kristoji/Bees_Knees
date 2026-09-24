"""Check the engine's draw handling against the printed rules and against real games.

The rules (Hive Pocket Ultimate, p.11 "The End of the Game") say:

    The person whose Queen Bee is surrounded loses the game, unless the last piece to
    surround their Queen Bee also completes the surrounding of the other Queen Bee, in
    that case the game is drawn. A draw may also be agreed if both players are in a
    position where they are forced to move the same two pieces over and over again.

So there are exactly two draws in the rules: the simultaneous surround, which is
automatic, and an AGREED draw on a forced shuffle, which no engine can detect on its
own. The engine adds a threefold-repetition draw, which is standard practice and keeps
a game from running forever.

This replays the recorded games and checks that the engine never ends one early: a
premature draw would mean the repetition rule is too eager, and a premature win would
mean the surround test is wrong.

    python bench/test_draw_rules.py --src data/raw
"""
import argparse
import collections
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from engine.board import Board  # noqa: E402
from engine.enums import GameState, PlayerColor  # noqa: E402

DECIDED = {"WhiteWins", "BlackWins", "Draw"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="directory of collection folders")
    parser.add_argument("--per-collection", type=int, default=300)
    args = parser.parse_args()

    agree = collections.Counter()
    early = collections.Counter()
    draw_kind = collections.Counter()
    total = 0

    for collection in sorted(os.listdir(args.src)):
        path = os.path.join(args.src, collection)
        if not os.path.isdir(path):
            continue
        for game in sorted(os.listdir(path))[:args.per_collection]:
            board_txt = os.path.join(path, game, "board.txt")
            if not os.path.exists(board_txt):
                continue
            parts = open(board_txt).read().splitlines()[0].strip().split(";")
            if len(parts) < 3 or parts[1] not in DECIDED:
                continue
            total += 1
            board = Board(parts[0] or "Base+MLP")
            stopped = None
            for i, move in enumerate(parts[3:]):
                try:
                    board.play(move)
                except Exception:
                    stopped = i
                    break
            if stopped is not None:
                early[f"{parts[1]} stopped at ply {stopped} as {board.state}"] += 1
                continue
            agree[f"{parts[1]} -> {board.state}"] += 1
            if parts[1] == "Draw":
                both = (board.count_queen_neighbors(PlayerColor.WHITE) == 6
                        and board.count_queen_neighbors(PlayerColor.BLACK) == 6)
                if both:
                    draw_kind["both queens surrounded (in the rules)"] += 1
                elif board.state is GameState.DRAW:
                    draw_kind["threefold repetition (engine addition)"] += 1
                else:
                    draw_kind["neither: an agreed draw the engine cannot see"] += 1

    print(f"replayed {total} games\n")
    print("recorded outcome -> engine's final state:")
    for key, count in sorted(agree.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<34} {count}")
    print("\nhow the drawn games ended:")
    for key, count in sorted(draw_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<46} {count}")
    print(f"\ngames the engine ended before the recording did: {sum(early.values())}")
    for key, count in early.items():
        print(f"  {key}: {count}")

    ok = not early
    print("\nOK" if ok else "\nFAILED: the engine ends games the recordings continue")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
