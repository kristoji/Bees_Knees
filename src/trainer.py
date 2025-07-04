# from ai.network import NeuralNetwork
from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState
from engineer import Engine
from engine.game import Move
from ai.oracle import Oracle
import numpy as np

engine = Engine()

def reset_log(string: str = ""):
    with open("test/log.txt", "w") as f:
        f.write(string)

reset_log()

f_theta = Oracle()
# Training Data: # Tuple of 3 np arrays: (in_mats, out_mats, values)
T = (np.array([]), np.array([]), np.array([]))  

number_of_iterations = 1
number_of_games = 5
number_of_rollouts = 1000      

for iteration in range(number_of_iterations):
    for game in range(number_of_games):
        engine.newgame(["Base+MLP"])
        s = engine.board
        T_game = []
        mcts_game = MCTS(oracle=f_theta, num_rollouts=number_of_rollouts)
        winner = None

        while not winner:

            print(s.turn, end=": ")
            mcts_game.run_simulation_from(s)

            pi: dict[Move, float] = mcts_game.get_moves_probs()
            T_game += Training.get_matrices_from_board(s, pi)
            
            a: str = mcts_game.action_selection(training=False)
            print(a)
            engine.play(a)
            winner: GameState = engine.board.state != GameState.IN_PROGRESS


        print("FINALMENTE BRUNO SBRUGNAMI")
        exit()
    #     for in_mat, out_mat in T_game:
    #         value: float = 1.0 if engine.board.state == GameState.WHITE_WIN else -1.0 if engine.board.state == GameState.BLACK_WIN else 0.0
    #         # T.append((in_mat, out_mat, value))
            
    #         # TODO: da testare !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #         shape = (Training.NUM_PIECES * Training.LAYERS, Training.SIZE, Training.SIZE)
    #         np.append(T[0], np.array(in_mat).reshape(shape), axis=0)
    #         np.append(T[1], np.array(out_mat).reshape(shape), axis=0)
    #         np.append(T[2], np.array(value).reshape(shape), axis=0)


    # f_theta_new: NeuralNetwork = f_theta.copy()
    # f_theta_new.training(T)

    # new_player = Player(game_rules, f_theta_new)
    # old_player = Player(game_rules, f_theta)
    # new_player_is_stronger = duel(new_player, old_player)
    
    # if new_player_is_stronger:
    #     f_theta = f_theta_new.copy()
    
#1: ruotare prima, T_game=[(in_mat, out_mat)], T=[](in, out, value)

