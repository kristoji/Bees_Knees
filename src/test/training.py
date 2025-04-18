
# idea 1
"""
piece sel -> {pieces, prob}
for piece, prov in pieces:
    move_sel ->  {move, prob}
    for move, prob in moves:
        tot_prob[move] = prob_piece_sel * prob_move_sel
        
ADV: una via di mezzo tra compatto e funzionante
DISADV: le 2 reti non sono sicuramente correlate
"""

# idea 2
"""
la rete toglie il 1 dal pezzo che vuole muovere e distribuisce la probabilità dove vuole muoverlo
ADV: più leggera
DISADV: se ci sono più pezzi uguali non riesce a capire quale muovere
"""

# idea 3 
"""
Un layer per ogni pezzo in modo che sia sempre riconoscibile la mossa
ADV: so se si muoverà wS1 o wS2
DISADV: ha un input enormeeeeee 
input: num_pezzi * layer(7) * 64 * 64
output: stessa dimensiona dell'input, in ogni cella probabilità di muovere il relativo pezzo in quella posizione 

"""






"""
ZerosumPerfinfGame
.initial_state()->s
.next_state(s, a)->s’
.game_phase(s)-> winner
"""
game_rules = ZeroSumPerfInfoGame()

"""
DeepNeuralNetwork
.__init__(game_rules)       # Base+MLP
.copy()
.training(T)
"""
f_theta = DeepNeuralNetwork(game_rules)

#Training Data
T = []

#Number of training iteration
number_of_iterations = 50

#Number of games per iteration
number_of_games = 1000

#Number of rollouts per MCTS simulation
number_of_rollouts = 50
        

for iteration in range(number_of_iterations):
    for game in range(number_of_games):
        s = game_rules.initial_state()
        T_game = []
        mcts_game = MCTS(f_theta, num_rollouts=number_of_rollouts)
        winner = None

        while not winner:
            turn += 1
            mcts_game.run_simulation_from(s)
            pi = mcts_game.get_moves_probs(s)
            Training.get_matrices(T_game, s, pi)
            a = action_selection(pi)
            s = game_rules.next_state(s, a)
            winner = game_rules.game_phase(s)

        for sample in T_game:
            s, pi = sample
            value = compute_value(winner, s)
            T = T + data_augmentation((s, pi, value))

    f_theta_new = f_theta.copy()
    f_theta_new.training(T)

    new_player = Player(game_rules, f_theta_new)
    old_player = Player(game_rules, f_theta)
    new_player_is_stronger = duel(new_player, old_player)
    
    if new_player_is_stronger:
        f_theta = f_theta_new.copy()

#1: ruotare prima, T_game=[(in_mat, out_mat)], T=[](in, out, value)

