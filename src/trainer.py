# from ai.network import NeuralNetwork
from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState
from engineer import Engine
from engine.game import Move
from ai.oracle import Oracle, OracleNN
import numpy as np
import os

    


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
os.makedirs("data", exist_ok=True)

ENGINE = Engine()

N_ITERATIONS = 1
N_GAMES = 50
N_DUELS = 10
N_ROLLOUTS = 1000   



def duel(new_player: Oracle, old_player: Oracle, games: int = 10) -> tuple[float, float]:
    """
    Duel between two players using MCTS to determine which player is stronger.
    Returns the number of wins for each player: old_player and new_player.
    """
    old_wins = 0
    new_wins = 0

    for game in games:

        ENGINE.newgame(["Base+MLP"])
        s = ENGINE.board
        mcts_game_old = MCTS(oracle=old_player, num_rollouts=N_ROLLOUTS)
        mcts_game_new = MCTS(oracle=new_player, num_rollouts=N_ROLLOUTS)

        white_player = mcts_game_old if game % 2 == 0 else mcts_game_new
        black_player = mcts_game_new if game % 2 == 0 else mcts_game_old


        while not winner:

            print(s.turn, end=": ")
            white_player.run_simulation_from(s, debug=False)
            a: str = mcts_game.action_selection(training=False)
            print(a)
            ENGINE.play(a)

            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS
            if winner:
                break

            print(s.turn, end=": ")
            black_player.run_simulation_from(s, debug=False)
            a: str = mcts_game.action_selection(training=False)
            print(a)
            ENGINE.play(a)

    
            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS


        if ENGINE.board.state == GameState.WHITE_WIN:
            old_wins += 1 if game % 2 == 0 else 0
            new_wins += 1 if game % 2 == 1 else 0
        elif ENGINE.board.state == GameState.BLACK_WIN:
            old_wins += 1 if game % 2 == 1 else 0
            new_wins += 1 if game % 2 == 0 else 0
        else:
            old_wins += 0.5
            new_wins += 0.5

    return old_wins, new_wins

def reset_log(string: str = ""):
    with open("test/log.txt", "w") as f:
        f.write(string)



def main(): 

    reset_log()
    
    f_theta = Oracle()
    cons_unsuccess = 0

    for iteration in range(N_ITERATIONS):
        os.mkdir("data/iteration_" + str(iteration), exist_ok=True)

        game = 0
        draw = 0
        while game < N_GAMES:

            T = (np.array([]), np.array([]), np.array([]))  # Initialize training data for this game
            T_game = []

            ENGINE.newgame(["Base+MLP"])
            s = ENGINE.board
            mcts_game = MCTS(oracle=f_theta, num_rollouts=N_ROLLOUTS)
            winner = None

            while not winner:

                print(s.turn, end=": ")
                mcts_game.run_simulation_from(s, debug=False)

                pi: dict[Move, float] = mcts_game.get_moves_probs()
                T_game += Training.get_matrices_from_board(s, pi)
                
                a: str = mcts_game.action_selection(training=True)
                print(a)
                ENGINE.play(a)
                winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS

            if ENGINE.board.state == GameState.DRAW:
                draw += 1
                if draw >= 0.2 * N_GAMES:
                    continue
            
            game += 1
            print(f"Game {game} finished with state {ENGINE.board.state.name}")

            for in_mat, out_mat in T_game:
                value: float = 1.0 if ENGINE.board.state == GameState.WHITE_WIN else -1.0 if ENGINE.board.state == GameState.BLACK_WIN else 0.0
                # T.append((in_mat, out_mat, value))
                
                # TODO: da testare !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                shape = (Training.NUM_PIECES * Training.LAYERS, Training.SIZE, Training.SIZE)
                np.append(T[0], np.array(in_mat).reshape(shape), axis=0)
                np.append(T[1], np.array(out_mat).reshape(shape), axis=0)
                # np.append(T[2], np.array(value).reshape(shape), axis=0)
                np.append(T[2], np.array(value).reshape((1,)), axis=0)

            # Save the training data for this game
            np.savez_compressed(
                f"data/iteration_{iteration}/game_{game}.npz",
                in_mats=T[0],
                out_mats=T[1],
                values=T[2]
            )

            
        T_total = (np.array([]), np.array([]), np.array([]))  # Initialize total training data
        # cicle over data/iteration_{iteration}
        for file in os.listdir(f"data/iteration_{iteration}"):
            if file.endswith(".npz"):
                data = np.load(f"data/iteration_{iteration}/{file}")
                in_mats = data['in_mats']
                out_mats = data['out_mats']
                values = data['values']
                # Append the data to the total training data
                T_total[0] = np.append(T_total[0], in_mats, axis=0)
                T_total[1] = np.append(T_total[1], out_mats, axis=0)
                T_total[2] = np.append(T_total[2], values, axis=0)


        if iteration == 0:
            f_theta_new: Oracle = OracleNN()
        else:
            f_theta_new: Oracle = f_theta.copy()

        f_theta_new.training(T_total)

        old_wins, new_wins = duel(f_theta_new, f_theta, games=N_DUELS)
        
        if old_wins < new_wins:
            f_theta = f_theta_new.copy()
            cons_unsuccess = 0
        else:
            cons_unsuccess += 1

        if cons_unsuccess >= 3:
            print(f"Stopping training after {iteration} iterations due to no improvement.")
            break
        

if "__main__" == __name__:
    main()
    print("Training completed.")