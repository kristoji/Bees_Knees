import subprocess
import random
from test.gpt import generate_match_graphs
import os

OK = "ok\n"

def send(p: subprocess.Popen, command: str) -> str:
    p.stdin.write(command + "\n")
    p.stdin.flush()
    return read_all(p)

def readuntil(p: subprocess.Popen, delim: str) -> str:
    output = []
    while True:
        line = p.stdout.readline()
        if not line:
            break
        output.append(line.strip())
        if line.endswith(delim):
            break
    return "\n".join(output)

def read_all(p: subprocess.Popen) -> str:
    return readuntil(p, OK)

def start_process(path) -> subprocess.Popen:
    return subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    )

def end_process(p: subprocess.Popen) -> None:
    p.stdin.close()
    p.stdout.close()
    p.stderr.close()
    p.kill()

def play_step(p1: subprocess.Popen, p2: subprocess.Popen, depth: int, time_sec: int) -> str:
    if depth > 0:
        move = send(p1, f"bestmove depth {depth}")
    else:
        h = time_sec // 3600
        m = (time_sec % 3600) // 60
        s = time_sec % 60
        move = send(p1, f"bestmove time {h:02}:{m:02}:{s:02}")
    move = move.strip().split("\n")[0]
    print(f"[Player] plays: {move}")
    send(p1, f"play {move}")
    return send(p2, f"play {move}")

def play_step_single_process(p1: subprocess.Popen, depth: int, time_sec: int) -> str:
    if depth > 0:
        move = send(p1, f"bestmove depth {depth}")
    else:
        h = time_sec // 3600
        m = (time_sec % 3600) // 60
        s = time_sec % 60
        move = send(p1, f"bestmove time {h:02}:{m:02}:{s:02}")
    move = move.strip().split("\n")[0]
    print(f"[Player] plays: {move}")
    return send(p1, f"play {move}")

def play_random_step_single_process(p1: subprocess.Popen) -> str:
    moves = send(p1, f"validmoves")
    moves_list = moves.strip().split("\n")[0].split(";")
    random_move = moves_list[random.randint(0, len(moves_list) - 1) ]
    print(f"[Player] plays: {random_move}")
    return send(p1, f"play {random_move}")

def check_end_game(out: str) -> bool:
    return "InProgress" != out.split(";")[1]

def run_games(engine_white_path: str, 
              engine_black_path: str, 
              num_games: int = 5, 
              max_turns: int = 50, 
              depth: int = 0, 
              time_sec: int = 2):
    assert depth > 0 or time_sec > 0, "Specify either depth or time_sec > 0"

    print(f"Starting engine 1 (White): {engine_white_path}")
    white = start_process(engine_white_path)
    read_all(white)

    print(f"Starting engine 2 (Black): {engine_black_path}")
    black = start_process(engine_black_path)
    read_all(black)

    white_wins = 0
    black_wins = 0
    draws = 0

    for game_id in range(num_games):
        print(f"\n--- Starting game {game_id + 1}/{num_games} ---")
        send(white, "newgame Base+MLP")
        send(black, "newgame Base+MLP")

        for _ in range(max_turns):
            out = play_step(white, black, depth, time_sec)
            if check_end_game(out):
                if "WhiteWins" in out:
                    white_wins += 1
                elif "BlackWins" in out:
                    black_wins += 1
                else:
                    draws += 1
                print(out)
                break

            out = play_step(black, white, depth, time_sec)
            if check_end_game(out):
                if "WhiteWins" in out:
                    white_wins += 1
                elif "BlackWins" in out:
                    black_wins += 1
                else:
                    draws += 1
                print(out)
                break

        engine_name_white = os.path.basename(engine_white_path)
        engine_name_black = os.path.basename(engine_black_path)
        game_dir = f"/root/Bees_Knees/data/{engine_name_white}-{engine_name_black}/game_{game_id + 1}"
        os.makedirs(game_dir, exist_ok=True)

        # save final board state
        final_txt = os.path.join(game_dir, f"board.txt")
        with open(final_txt, 'w') as f_out:
            f_out.write(out)

        moves = out_to_moves(out)
        generate_match_graphs(game_dir=game_dir, game_moves=moves)

    send(white, "exit")
    send(black, "exit")
    end_process(white)
    end_process(black)

    print("\n========== FINAL RESULTS ==========")
    print(f"White wins: {white_wins}")
    print(f"Black wins: {black_wins}")
    print(f"Draws     : {draws}")

def duel_random(engine_path: str, 
                num_games: int = 5, 
                max_turns: int = 50, 
                depth: int = 0, 
                time_sec: int = 2):
    assert depth > 0 or time_sec > 0, "Specify either depth or time_sec > 0"

    print(f"Starting engine 1 (White): {engine_path}")
    engine = start_process(engine_path)
    read_all(engine)

    engine_wins = 0
    random_wins = 0
    draws = 0

    for game_id in range(num_games):
        print(f"\n--- Starting game {game_id + 1}/{num_games} ---")
        send(engine, "newgame Base+MLP")

        for _ in range(max_turns):
            out = play_random_step_single_process(engine)
            if check_end_game(out):
                if "WhiteWins" in out:
                    random_wins += 1
                elif "BlackWins" in out:
                    engine_wins += 1
                else:
                    draws += 1
                print(out)
                break

            out = play_step_single_process(engine, depth, time_sec)
            if check_end_game(out):
                if "WhiteWins" in out:
                    random_wins += 1
                elif "BlackWins" in out:
                    engine_wins += 1
                else:
                    draws += 1
                print(out)
                break

        engine_name = os.path.basename(engine_path)
        game_dir = f"/root/Bees_Knees/data/random-{engine_name}/game_{game_id + 1}"
        os.makedirs(game_dir, exist_ok=True)

        # save final board state
        final_txt = os.path.join(game_dir, f"board.txt")
        with open(final_txt, 'w') as f_out:
            f_out.write(out)

        moves = out_to_moves(out)
        generate_match_graphs(game_dir=game_dir, game_moves=moves)

    send(engine, "exit")
    end_process(engine)

    print("\n========== FINAL RESULTS ==========")
    print(f"White wins: {engine_wins}")
    print(f"Black wins: {random_wins}")
    print(f"Draws     : {draws}")

def out_to_moves(out: str) -> list:
    moves = out.split("\n")[0].split(";")[3:]
    return moves

# Main di prova
def main():
    # Example usage
    # duel_random("/root/Bees_Knees/src/duel/nokamute", num_games=3, max_turns=50, depth=0, time_sec=2)
    run_games(
       engine_white_path="/root/Bees_Knees/src/duel/nokamute",
       engine_black_path="/root/Bees_Knees/src/duel/nokamute",
       num_games=3,
       max_turns=50,
       depth=0,
       time_sec=6
    )
if __name__ == "__main__":
    main()