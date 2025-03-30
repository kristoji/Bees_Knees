"""
ZerosumPerfinfGame
.initial_state()->s
.next_state(s, a)->s’
.game_phase(s)-> winner
"""
game_rules = ZerosumPerfinfGame()

"""
DeepNeuralNetwork
.__init__(game_rules)
.copy()
.training(T)
"""
f_theta = DeepNeuralNetwork(game_rules)

#Training Data
T = []

#Number of training iteration
number_of_iteration = 50

#Number of games per iteration
number_of_games = 1000

for iteration in range(number_of_iteration):

    for game in range(number_of_games):

        s = game_rules.initial_state() 

        T_game = []

        mcts_game = MCTS(game_rules, f_theta)

        turn = 0
        winner = None

        while not winner:
            
