
from ai.brains import MCTS, Random
from engine.enums import GameState
from ai.oracle import Oracle
from ai.log_utils import log_header, log_subheader
from engineer import Engine
from test.subp import start_process, read_all, send, play_step_single_process, check_end_game
from engine.board import Board
import subprocess


N_DUELS = 10                # Number of games to play in a duel
VERBOSE = True              # If True, prints the board state after each move
TIME_LIMIT = 5.0            # seconds for each MCTS simulation
DRAW_LIMIT = 100            # turns after the match ends in a draw


def duel_random(    player: Oracle, 
                    games: int = N_DUELS, 
                    time_limit:float = TIME_LIMIT, 
                    verbose:bool = VERBOSE, 
                    draw_limit:int = DRAW_LIMIT) -> tuple[float, float]:
    """
    Duel between two players using MCTS to determine which player is stronger.
    Returns the number of wins for each player: old_player and new_player.
    """
    engine = Engine()
    player_wins = 0
    random_wins = 0

    for game in range(games):

        log_subheader(f"Duel Game {game + 1} of {games}: PLAYER {player_wins} - {random_wins} RANDOM")

        engine.newgame(["Base+MLP"])
        s = engine.board
        winner = None

        mcts_player = MCTS(oracle=player, time_limit=time_limit, debug=True)
        random_player = Random()

        white_player = mcts_player if game % 2 == 0 else random_player
        black_player = random_player if game % 2 == 0 else mcts_player

        i = 0
        while not winner:
            
            a:str = white_player.calculate_best_move(s, "time", time_limit)
            engine.play(a, verbose=verbose)
            i+= 1

            winner: GameState = engine.board.state != GameState.IN_PROGRESS

            if winner:
                break

            if i == draw_limit:
                winner: GameState = GameState.DRAW
                break
            
            a: str = black_player.calculate_best_move(s, "time", time_limit)
            engine.play(a, verbose=verbose)
            i+= 1
    
            winner: GameState = engine.board.state != GameState.IN_PROGRESS

            if winner:
                break

            if i == draw_limit:
                winner: GameState = GameState.DRAW
                break

            print("", end="", flush=True)


        if engine.board.state == GameState.WHITE_WINS:
            player_wins += 1 if game % 2 == 0 else 0
            random_wins += 1 if game % 2 == 1 else 0
        elif engine.board.state == GameState.BLACK_WINS:
            player_wins += 1 if game % 2 == 1 else 0
            random_wins += 1 if game % 2 == 0 else 0
        else:
            player_wins += 0.5
            random_wins += 0.5

    log_subheader(f"End of Duel: PLAYER {player_wins} - {random_wins} RANDOM")

    return player_wins, random_wins

def duel(   new_player: Oracle, 
            old_player: Oracle, 
            games: int = N_DUELS, 
            time_limit:int = TIME_LIMIT, 
            verbose:bool = VERBOSE, 
            draw_limit:int = DRAW_LIMIT) -> tuple[float, float]:
    """
    Duel between two players using MCTS to determine which player is stronger.
    Returns the number of wins for each player: old_player and new_player.
    """
    engine = Engine()
    old_wins = 0
    new_wins = 0

    for game in range(games):

        log_subheader(f"Duel Game {game + 1} of {games}: OLD {old_wins} - {new_wins} NEW")

        engine.newgame(["Base+MLP"])
        s = engine.board
        winner = None

        mcts_game_old = MCTS(oracle=old_player, time_limit=time_limit)
        mcts_game_new = MCTS(oracle=new_player, time_limit=time_limit)

        white_player = mcts_game_old if game % 2 == 0 else mcts_game_new
        black_player = mcts_game_new if game % 2 == 0 else mcts_game_old

        i = 0
        while not winner:

            a: str = white_player.calculate_best_move(s, "time", time_limit)
            engine.play(a, verbose=verbose)
            i+= 1

            winner: GameState = engine.board.state != GameState.IN_PROGRESS

            if winner:
                break

            if i == draw_limit:
                winner: GameState = GameState.DRAW
                break

            a: str = black_player.calculate_best_move(s, "time", time_limit)
            engine.play(a, verbose=verbose)
            i+= 1
    
            winner: GameState = engine.board.state != GameState.IN_PROGRESS

            if winner:
                break

            if i == draw_limit:
                winner: GameState = GameState.DRAW
                break


        if engine.board.state == GameState.WHITE_WINS:
            old_wins += 1 if game % 2 == 0 else 0
            new_wins += 1 if game % 2 == 1 else 0
        elif engine.board.state == GameState.BLACK_WINS:
            old_wins += 1 if game % 2 == 1 else 0
            new_wins += 1 if game % 2 == 0 else 0
        else:
            old_wins += 0.5
            new_wins += 0.5

    return old_wins, new_wins



def cross_platform_duel(    exe_player_path: str,
                            oracle_player: Oracle, 
                            is_exe_white:bool,
                            games: int = N_DUELS, 
                            time_limit:int = TIME_LIMIT, 
                            verbose:bool = VERBOSE, 
                            draw_limit:int = DRAW_LIMIT) -> tuple[float, float]:
    """
    Duel between two players using MCTS to determine which player is stronger.
    Returns the number of wins for each player: old_player and new_player.
    """

    def play_exe(engine: Engine, exe_player: subprocess.Popen, s: Board, time_limit: int, verbose: bool, i: int):
        out = play_step_single_process(exe_player, depth=0, time_sec=time_limit)
        out_move = out.split("\n")[0].split(";")[-1]
        engine.play(out_move, verbose=verbose)
        if i+1 == draw_limit:
            return GameState.DRAW
            
        return engine.board.state

    def play_mcts(engine: Engine, mcts_player: MCTS, exe_player: subprocess.Popen, s: Board, time_limit: int, verbose: bool, i:int):
        a: str = mcts_player.calculate_best_move(s, "time", time_limit)
        engine.play(a, verbose=verbose)
        send(exe_player, f"play {a}")
        if i+1 == draw_limit:
            return GameState.DRAW
        return engine.board.state

    engine = Engine()
    exe_wins = 0
    oracle_wins = 0



    print(f"Starting engine 1: {exe_player_path}")
    exe_player = start_process(exe_player_path)
    read_all(exe_player)

    for game in range(games):

        log_subheader(f"Duel Game {game + 1} of {games} with white as {'EXE' if is_exe_white else 'ORACLE'}: EXE {exe_wins} - {oracle_wins} ORACLE")

        engine.newgame(["Base+MLP"])
        s = engine.board

        mcts_player = MCTS(oracle=oracle_player, time_limit=time_limit)
        send(exe_player, "newgame Base+MLP")

        i = 0
        while True:


            if is_exe_white:

                state = play_exe(engine, exe_player, s, time_limit, verbose, i)
                i += 1
                if state != GameState.IN_PROGRESS:
                    break

                state = play_mcts(engine, mcts_player, exe_player, s, time_limit, verbose, i)
                i += 1
                if state != GameState.IN_PROGRESS:
                    break

            else:

                state = play_mcts(engine, mcts_player, exe_player, s, time_limit, verbose, i)
                i += 1
                if state != GameState.IN_PROGRESS:
                    break

                state = play_exe(engine, exe_player, s, time_limit, verbose, i)
                i += 1
                if state != GameState.IN_PROGRESS:
                    break

            print("", end="", flush=True)


        if state == GameState.WHITE_WINS:
            exe_wins += 1 if is_exe_white else 0
            oracle_wins += 1 if not is_exe_white else 0
        elif state == GameState.BLACK_WINS:
            exe_wins += 1 if not is_exe_white else 0
            oracle_wins += 1 if is_exe_white else 0
        else:
            exe_wins += 0.5
            oracle_wins += 0.5

        is_exe_white = not is_exe_white  # Switch sides for the next game

    log_subheader(f"End of Duel: EXE {exe_wins} - {oracle_wins} ORACLE")

    return exe_wins, oracle_wins