# from ai.network import NeuralNetwork
from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState
from engineer import Engine
from engine.game import Move
from ai.oracle import Oracle, OracleNN
import numpy as np
import os
from datetime import datetime

    


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if BASE_PATH[-3:] == "src":
    BASE_PATH = BASE_PATH[:-3]
print(BASE_PATH)
os.chdir(BASE_PATH)  # Change working directory to the base path
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# GLOBALS
ENGINE = Engine()

# PARAMS
N_ITERATIONS = 1
N_GAMES = 5
N_DUELS = 10
N_ROLLOUTS = 1000
ALLOW_DRAWS = 3
VERBOSE = True
# DEBUG = False


def duel(new_player: Oracle, old_player: Oracle, games: int = 10) -> tuple[float, float]:
    """
    Duel between two players using MCTS to determine which player is stronger.
    Returns the number of wins for each player: old_player and new_player.
    """
    old_wins = 0
    new_wins = 0

    for game in range(games):

        log_subheader(f"Duel Game {game + 1} of {games}: OLD {old_wins} - {new_wins} NEW")

        ENGINE.newgame(["Base+MLP"])
        s = ENGINE.board
        winner = None

        mcts_game_old = MCTS(oracle=old_player, num_rollouts=N_ROLLOUTS)
        mcts_game_new = MCTS(oracle=new_player, num_rollouts=N_ROLLOUTS)

        white_player = mcts_game_old if game % 2 == 0 else mcts_game_new
        black_player = mcts_game_new if game % 2 == 0 else mcts_game_old


        while not winner:

            # print(s.turn, end=": ")
            white_player.run_simulation_from(s, debug=False)
            a: str = white_player.action_selection(training=False)
            # print(a)
            ENGINE.play(a, verbose=VERBOSE)

            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS
            if winner:
                break

            # print(s.turn, end=": ")
            black_player.run_simulation_from(s, debug=False)
            a: str = black_player.action_selection(training=False)
            # print(a)
            ENGINE.play(a, verbose=VERBOSE)

    
            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS


        if ENGINE.board.state == GameState.WHITE_WINS:
            old_wins += 1 if game % 2 == 0 else 0
            new_wins += 1 if game % 2 == 1 else 0
        elif ENGINE.board.state == GameState.BLACK_WINS:
            old_wins += 1 if game % 2 == 1 else 0
            new_wins += 1 if game % 2 == 0 else 0
        else:
            old_wins += 0.5
            new_wins += 0.5

    return old_wins, new_wins

def reset_log(string: str = ""):
    return
    with open("test/log.txt", "w") as f:
        f.write(string)

def log_header(title: str, width: int = 60, char: str = '='):
    bar = char * width
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{bar}")
    print(f"{ts} | {title.center(width - len(ts) - 3)}")
    print(f"{bar}\n")

def log_subheader(title: str, width: int = 50, char: str = '-'):
    bar = char * width
    print(f"{bar}")
    print(f"{title.center(width)}")
    print(f"{bar}")


def main(): 

    reset_log()
    
    f_theta = Oracle()
    cons_unsuccess = 0

    for iteration in range(N_ITERATIONS):
        os.makedirs("data/iteration_" + str(iteration), exist_ok=True)

        log_header(f"STARTING ITERATION {iteration}")

        game = 0
        draw = 0
        while game < N_GAMES:

            log_subheader(f"Game {game} (Draws so far: {draw})")

            T_game = []

            ENGINE.newgame(["Base+MLP"])
            s = ENGINE.board
            mcts_game = MCTS(oracle=f_theta, num_rollouts=N_ROLLOUTS)
            winner = None

            while not winner:

                # print(s.turn, end=": ")
                mcts_game.run_simulation_from(s, debug=False)

                pi: dict[Move, float] = mcts_game.get_moves_probs()
                T_game += Training.get_matrices_from_board(s, pi)
                
                a: str = mcts_game.action_selection(training=True)
                # print(a)
                ENGINE.play(a, verbose=VERBOSE)
                winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS
                # winner = True #[DBG]

            if ENGINE.board.state == GameState.DRAW:
                draw += 1
                if draw > ALLOW_DRAWS * N_GAMES:
                    continue
            
            game += 1
            print(f"Game {game} finished with state {ENGINE.board.state.name}")

            value: float = 1.0 if ENGINE.board.state == GameState.WHITE_WINS else -1.0 if ENGINE.board.state == GameState.BLACK_WINS else 0.0
            # value = 1.0 #[DBG]

            game_shape = (0, *Training.INPUT_SHAPE)
            T_0 = np.empty(shape=game_shape, dtype=np.float32)
            T_1 = np.empty(shape=game_shape, dtype=np.float32)
            T_2 = np.empty(shape=(0,), dtype=np.float32)

            for in_mat, out_mat in T_game:
                T_2 = np.append(T_2, np.array(value, dtype=np.float32).reshape((1,)), axis=0)
                T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
                T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

            # Save the training data for this game
            np.savez_compressed(
                f"data/iteration_{iteration}/game_{game}.npz",
                in_mats=T_0,
                out_mats=T_1,
                values=T_2,
            )

        it_shape = (0, *Training.INPUT_SHAPE)
        Ttot_0, Ttot_1, Ttot_2 = (np.empty(shape=it_shape, dtype=np.float32), np.empty(shape=it_shape, dtype=np.float32), np.empty(shape=(0,), dtype=np.float32))

        for file in os.listdir(f"data/iteration_{iteration}"):
            if file.endswith(".npz"):
                data = np.load(f"data/iteration_{iteration}/{file}", allow_pickle=True)
                in_mats = np.array(data['in_mats'], dtype=np.float32)
                out_mats = np.array(data['out_mats'], dtype=np.float32)
                values = np.array(data['values'], dtype=np.float32)
                # Append the data to the total training data
                Ttot_0 = np.append(Ttot_0, in_mats, axis=0)
                Ttot_1 = np.append(Ttot_1, out_mats, axis=0)
                Ttot_2 = np.append(Ttot_2, values, axis=0)


        if iteration == 0:
            f_theta_new: Oracle = OracleNN()
        else:
            f_theta_new: Oracle = f_theta.copy()

        f_theta_new.training((Ttot_0, Ttot_1, Ttot_2))

        log_header("STARTING DUEL")

        old_wins, new_wins = duel(f_theta_new, f_theta, games=N_DUELS)
        
        if old_wins < new_wins:
            f_theta = f_theta_new.copy()
            f_theta.save(f"models/{iteration}.npz")
            cons_unsuccess = 0
        else:
            cons_unsuccess += 1

        if cons_unsuccess >= 3:
            print(f"Stopping training after {iteration} iterations due to no improvement.")
            break
        

if "__main__" == __name__:
    main()
    print("Training completed.")