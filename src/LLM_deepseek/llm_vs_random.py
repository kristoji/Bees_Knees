from typing import Optional, Dict, Any
from math import ceil

import sys
from pathlib import Path

# Ensure the `src` folder is on sys.path so internal top-level imports like
# `engine` and `ai` resolve whether this file is run as a script or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engineer import Engine
from engine.enums import GameState
from ai.brains import Random
from ai.llm_brain import LLMSequentialBrain


def duel_llm_vs_random(
    games: int = 10,
    verbose: bool = False,
    draw_limit: int = 150,
    model_dir: Optional[str] = None,
    sequence_length: int = 5,
    show_moves: bool = False,
) -> Dict[str, Any]:
    """
    Play a duel between the LLMSequentialBrain and the Random brain for `games` matches.
    Alternates colors each game. Returns aggregate metrics including win-rate and
    moves-to-win statistics.
    """

    engine = Engine()

    llm_brain = LLMSequentialBrain(model_dir=model_dir, sequence_length=sequence_length)
    rnd_brain = Random()

    llm_points = 0.0
    rnd_points = 0.0
    draws = 0

    moves_to_llm_win_full = []   # full moves (plies//2 rounded up)
    moves_to_llm_win_plies = []  # raw plies (Board.turn)

    for game in range(games):
        engine.newgame(["Base+MLP"])  # set expansion set
        i = 0  # plies counter

        # Sides alternate
        white_player = llm_brain if game % 2 == 0 else rnd_brain
        black_player = rnd_brain if game % 2 == 0 else llm_brain

        if show_moves:
            print(f"\nGame {game + 1}/{games} — White: {'LLM' if isinstance(white_player, LLMSequentialBrain) else 'Random'} | "
                  f"Black: {'LLM' if isinstance(black_player, LLMSequentialBrain) else 'Random'}")

        # Clear any cached move in brains
        white_player.empty_cache()
        black_player.empty_cache()

        # Play until terminal or draw limit
        while True:
            # White plays
            a = white_player.calculate_best_move(engine.board, restriction="depth", value=0)
            a_str = a if isinstance(a, str) else engine.board.stringify_move(a)
            if show_moves:
                side_label = "LLM" if isinstance(white_player, LLMSequentialBrain) else "Random"
                print(f"Ply {i + 1:>3} | White-{side_label}: {a_str}")
            engine.play(a_str, verbose=verbose)
            i += 1

            if engine.board.state != GameState.IN_PROGRESS:
                break
            if i >= draw_limit:
                engine.board.state = GameState.DRAW
                break

            # Black plays
            a = black_player.calculate_best_move(engine.board, restriction="depth", value=0)
            a_str = a if isinstance(a, str) else engine.board.stringify_move(a)
            if show_moves:
                side_label = "LLM" if isinstance(black_player, LLMSequentialBrain) else "Random"
                print(f"Ply {i + 1:>3} | Black-{side_label}: {a_str}")
            engine.play(a_str, verbose=verbose)
            i += 1

            if engine.board.state != GameState.IN_PROGRESS:
                break
            if i >= draw_limit:
                engine.board.state = GameState.DRAW
                break

        # Score and metrics
        if engine.board.state == GameState.WHITE_WINS:
            if isinstance(white_player, LLMSequentialBrain):
                llm_points += 1.0
                moves_to_llm_win_plies.append(i)
                moves_to_llm_win_full.append(ceil(i / 2))
            else:
                rnd_points += 1.0
        elif engine.board.state == GameState.BLACK_WINS:
            if isinstance(black_player, LLMSequentialBrain):
                llm_points += 1.0
                moves_to_llm_win_plies.append(i)
                moves_to_llm_win_full.append(ceil(i / 2))
            else:
                rnd_points += 1.0
        else:
            draws += 1
            llm_points += 0.5
            rnd_points += 0.5

    total = games
    result = {
        "games": total,
        "llm_points": llm_points,
        "random_points": rnd_points,
        "draws": draws,
        "llm_win_rate": llm_points / total,
        "llm_wins_count": int(llm_points - 0.5 * draws),
        "moves_to_llm_win_full": moves_to_llm_win_full,
        "moves_to_llm_win_plies": moves_to_llm_win_plies,
        "avg_full_moves_to_win": sum(moves_to_llm_win_full) / len(moves_to_llm_win_full) if moves_to_llm_win_full else None,
        "avg_plies_to_win": sum(moves_to_llm_win_plies) / len(moves_to_llm_win_plies) if moves_to_llm_win_plies else None,
    }

    return result


if __name__ == "__main__":
    # Lightweight CLI run
    import argparse
    import json

    parser = argparse.ArgumentParser(description="LLM vs Random duel metrics")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--draw_limit", type=int, default=150)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--show_moves", action="store_true", help="Print each move with side and brain type")

    args = parser.parse_args()

    metrics = duel_llm_vs_random(
        games=args.games,
        verbose=args.verbose,
        draw_limit=args.draw_limit,
        model_dir=args.model_dir,
        sequence_length=args.sequence_length,
    show_moves=args.show_moves,
    )

    print(json.dumps(metrics, indent=2))
