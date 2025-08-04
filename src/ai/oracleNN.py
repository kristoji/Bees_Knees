from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np
from ai.brains import MCTS
from ai.network import NeuralNetwork
from ai.training import Training
from ai.oracle import Oracle
from engineer import Engine
import os
from trainer import log_header, log_subheader
from engine.enums import GameState

SHORT = 50
LONG = 100
SUPERLONG = 300

class OracleNN(Oracle):
    """
    Oracle that uses a neural network to predict the value and policy of a board state.
    """
    def __init__(self):
        self.network = NeuralNetwork()
        self.path = ""    

    def training(self, train_data_path:str, epochs:int) -> None:
        """
        Train the neural network with the provided training data.
        T is a tuple of (in_mats, out_mats, values)
        """
        if not self.network:
            raise ValueError("Neural network is not initialized.")
        self.network.train_network(
            train_data_path = train_data_path,
            num_epochs=epochs, 
            batch_size=32, 
            learning_rate=0.001,
            value_loss_weight=0.5 
        )

    def save(self, path: str) -> None:
        """
        Save weights
        """
        self.path = path
        self.network.save(path)


    def copy(self) -> 'OracleNN':
        """
        Create a copy of the Oracle instance.
        """
        if not self.path:
            # self.save("temp.pth") # save in a temp file just to perform the copy
            raise ValueError("Path is not set. Cannot copy without a path.")
        new_oracle = OracleNN()
        new_oracle.network.load(self.path) 
        return new_oracle
    
    def compute_heuristic(self, board) -> float:
        v, _ = self.predict(board)
        return v

    def predict(self, board: Board) -> tuple[float, Dict[Move, float]]:
        """
        Predict the value and policy for the given board state.
        """
        T = Training.get_in_mat_from_board(board)
        T = np.array(T, dtype=np.float32).reshape((1, *Training.INPUT_SHAPE))
        v, pi_mat = self.network.predict(T)

        # print(pi_mat.shape) # (1, 109760)

        pi = Training.get_dict_from_matrix(pi_mat[0], board)
        valid_moves = list(board.get_valid_moves())
        # Filter pi to only include valid moves
        pi = {move: prob for move, prob in pi.items() if move in valid_moves}
        # Softmax the probabilities
        if pi:
            probs = np.array(list(pi.values()))
            probs = np.exp(probs - np.max(probs))
            probs /= np.sum(probs)
            pi = {move: prob for move, prob in zip(pi.keys(), probs)}
        # else:
        #     raise ValueError("No valid moves found in the board state.")

        return v, pi
    
    def generate_matches(self, iteration_folder: str, n_games: int = 500, n_rollouts: int = 1500, verbose: bool = False, perc_allowed_draws: float = float('inf')) -> None:

        engine = Engine()
        os.makedirs(iteration_folder, exist_ok=True)

        game = 1
        draw = 0
        wins = 0
        discarded = 0

        while game < n_games:

            log_header(f"Game {game} of {n_games}: {draw}/{wins} [D/W] - Discarded: {discarded}", width=70)

            T_game = []
            v_values = []

            engine.newgame(["Base+MLP"])
            s = engine.board
            mcts_game = MCTS(oracle=self, num_rollouts=n_rollouts)
            winner = None

            num_moves = 0
            value = 1.0

            while not winner and num_moves < SUPERLONG:

                mcts_game.run_simulation_from(s, debug=False)

                pi: dict[Move, float] = mcts_game.get_moves_probs()

                mats = Training.get_matrices_from_board(s, pi)
                v_values += [value]*len(mats)
                value *= -1.0
                T_game += mats

                a: str = mcts_game.action_selection(training=True)

                engine.play(a, verbose=verbose)

                winner: GameState = engine.board.state != GameState.IN_PROGRESS

                num_moves += 1

            if num_moves >= SUPERLONG: # to avoid killing process for cache overflow
                log_subheader(f"Game {game} exceeded maximum moves ({SUPERLONG}). Ending game early.")
                discarded += 1
                continue
            elif engine.board.state == GameState.DRAW:
                draw += 1
                if draw > perc_allowed_draws * n_games:
                    continue
            else:
                wins += 1

            log_subheader(f"Game {game} finished with state {engine.board.state.name}")

            final_value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0

            # -------------- 1st VERSION  --------------
            game_shape = (0, *Training.INPUT_SHAPE)
            T_0 = np.empty(shape=game_shape, dtype=np.float32)
            T_1 = np.empty(shape=game_shape, dtype=np.float32)
            T_2 = np.array(v_values, dtype=np.float32)
            T_2 = T_2 * final_value # update the values using final_value


            for in_mat, out_mat in T_game:
                T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
                T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

            win_or_draw = "draws" if engine.board.state == GameState.DRAW else "wins"
            short_or_long = "short" if num_moves < SHORT else "long" if num_moves < LONG else "superlong"

            # Create the subdirectories for saving
            save_dir = f"{iteration_folder}/{win_or_draw}/{short_or_long}"
            os.makedirs(save_dir, exist_ok=True)

            log_subheader(f"Saving game {game} in {save_dir}/game_{game}.npz")

            # Save the training data for this game
            np.savez_compressed(
                f"{save_dir}/game_{game}.npz",
                in_mats=T_0,
                out_mats=T_1,
                values=T_2,
            )

            log_subheader(f"Saving board state in {save_dir}/board_{game}.txt")

            with open(f"{save_dir}/board_{game}.txt", "w") as f:
                f.write(str(engine.board))

            game += 1

