"""Play the trained agent against an external UHP engine, under a shared time control.

Beating the built-in heuristic says the network learned something real. It does not say
the agent is strong, because the heuristic only counts queen neighbours. This drives a
real engine over the Universal Hive Protocol instead.

Both sides get the same wall-clock budget per move, which is the only honest way to
compare two searches with different costs per node: "1600 rollouts against 2 seconds"
would compare nothing.

    python tools/duel_uhp.py --engine ../hive/Mzinga.MacOSArm64/MzingaEngine \
        --weights models/agent-71879/best.pt --summary models/agent-71879/summary.json \
        --games 10 --move-time 2

The external engine's own strength setting matters: Mzinga plays at whatever level its
time control allows, so the result is a statement about this time control, not about the
engines in general.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ai.mcts_batch import MCTS_BATCH  # noqa: E402
from engine.board import Board  # noqa: E402
from engine.enums import GameState, PlayerColor  # noqa: E402

# Minimal UHP client. test/subp.py has one, but importing it pulls in test.gpt and with
# it networkx and matplotlib, which this has no use for.
OK = "ok"


def start_engine(path):
    proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    # A UHP engine greets with its id and capabilities followed by "ok". Leaving that
    # unread shifts every later reply by one command.
    banner = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{path} exited before greeting")
        line = line.strip()
        if line == OK:
            return proc, banner
        banner.append(line)


def send(proc, command):
    """Send one command and collect the reply up to the engine's 'ok' line."""
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"the engine died on: {command}")
        line = line.strip()
        if line == OK:
            return lines
        if line.startswith("err") or line.startswith("invalidmove"):
            raise RuntimeError(f"engine rejected '{command}': {line}")
        lines.append(line)


def as_clock(seconds):
    return f"{int(seconds) // 3600:02d}:{int(seconds) % 3600 // 60:02d}:{seconds % 60:05.2f}"


def load_agent(weights, summary_path, device):
    from ai.oracleGNN import OracleGNN
    import torch

    cfg = json.load(open(summary_path))["args"]
    kwargs = dict(hidden_dim=cfg["hidden_dim"], conv_type=cfg["conv_type"],
                  num_layers=cfg["num_layers"], gat_heads=cfg["gat_heads"],
                  conv_dropout=cfg["dropout"], mlp_dropout=cfg["dropout"],
                  final_dropout=cfg["dropout"], use_layer_norm=True, use_residual=False,
                  pooling=cfg["pooling"], mlp_layers=2, final_mlp_layers=2)
    oracle = OracleGNN(device=device, **kwargs)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if any(k.startswith("policy.") for k in state):
        oracle.load_agent(weights, **kwargs)
        cfg["policy_head"] = True
    else:
        oracle.load(weights)
        cfg["policy_head"] = False
    return oracle, cfg


def random_opening(board, plies, seed):
    """Deterministic searches would replay one game, so the variety starts here."""
    rng = random.Random(seed)
    moves = []
    for _ in range(plies):
        legal = sorted(board.get_valid_moves(), key=board.stringify_move)
        if not legal:
            break
        chosen = legal[rng.randrange(len(legal))]
        moves.append(board.stringify_move(chosen))
        board.play(moves[-1])
    return moves


def play_game(oracle, proc, agent_is_white, move_time, max_plies, exploration,
              opening_plies, opening_seed, verbose):
    board = Board("Base+MLP")
    opening = random_opening(board, opening_plies, opening_seed)
    send(proc, "newgame Base+MLP")
    for move in opening:
        send(proc, f"play {move}")

    agent = MCTS_BATCH(oracle=oracle, exploration_weight=exploration,
                       num_rollouts=1, time_limit=move_time)
    agent_colour = PlayerColor.WHITE if agent_is_white else PlayerColor.BLACK
    rollouts = []

    while board.turn < max_plies:
        our_turn = board.current_player_color is agent_colour
        if our_turn:
            move = agent.calculate_best_move(board, restriction="time", value=move_time)
            rollouts.append(agent.last_rollouts)
            board.play(move)
            send(proc, f"play {move}")
        else:
            reply = send(proc, f"bestmove time {as_clock(move_time)}")
            move = reply[0].strip()
            # bestmove only computes; the engine does not apply it to its own board.
            # Without this the two boards drift apart by one ply and the engine starts
            # rejecting our moves as "not that player's turn".
            send(proc, f"play {move}")
            board.play(move)
        if verbose:
            print(f"      {'agent' if our_turn else 'engine'}: {move}")
        if board.state is not GameState.IN_PROGRESS:
            return board.state, board.turn, rollouts
    return None, board.turn, rollouts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, help="path to the UHP engine binary")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--move-time", type=float, default=2.0,
                        help="seconds per move, the same for both sides")
    parser.add_argument("--max-plies", type=int, default=100,
                        help="reaching this many plies is scored as a draw")
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--exploration", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    oracle, cfg = load_agent(args.weights, args.summary, args.device)
    proc, ident = start_engine(args.engine)
    print(f"opponent : {' / '.join(ident)}")
    print(f"agent    : {cfg['conv_type']} x{cfg['num_layers']}, hidden {cfg['hidden_dim']}, "
          f"{'policy head' if cfg.get('policy_head') else 'value only'}")
    print(f"control  : {args.move_time}s per move for both sides, "
          f"{args.max_plies}-ply cap, colours alternate\n")

    wins = losses = draws = capped = 0
    all_rollouts = []
    start = time.perf_counter()
    try:
        for game in range(args.start_game, args.start_game + args.games):
            agent_is_white = game % 2 == 0
            state, plies, rollouts = play_game(
                oracle, proc, agent_is_white, args.move_time, args.max_plies,
                args.exploration, args.opening_plies, args.seed + game // 2, args.verbose)
            all_rollouts += rollouts
            if state is None:
                capped += 1
                result = "cap"
            elif state is GameState.DRAW:
                draws += 1
                result = "draw"
            else:
                agent_won = (state is GameState.WHITE_WINS) == agent_is_white
                wins += agent_won
                losses += not agent_won
                result = "agent wins" if agent_won else "engine wins"
            print(f"  game {game:3d}  agent as {'White' if agent_is_white else 'Black'}  "
                  f"{plies:3d} plies  {result}", flush=True)
    finally:
        try:
            send(proc, "exit")
        except Exception:
            pass
        proc.kill()

    total = args.games
    score = (wins + 0.5 * (draws + capped)) / max(1, total)
    elapsed = time.perf_counter() - start
    mean_rollouts = sum(all_rollouts) / max(1, len(all_rollouts))
    print(f"\nagent {wins} - {losses} engine, {draws + capped} draws "
          f"({draws} on the board, {capped} at the {args.max_plies}-ply cap)")
    print(f"score {score * 100:.1f}%   agent averaged {mean_rollouts:.0f} rollouts "
          f"per move in {args.move_time}s")
    print(f"{elapsed:.0f}s, {elapsed / max(1, total):.0f}s per game")

    if args.out:
        json.dump({"wins": wins, "losses": losses, "draws": draws, "capped": capped,
                   "score": score, "games": total, "move_time": args.move_time,
                   "mean_rollouts": mean_rollouts, "opponent": ident,
                   "weights": args.weights, "seconds": elapsed},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
