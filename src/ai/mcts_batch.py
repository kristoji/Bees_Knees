from ai.brains import Brain
import math
from random import choice, uniform
from engine.board import Board
from engine.enums import GameState, PlayerColor, Error
from engine.game import Move
from time import time
from ai.oracle import Oracle
from ai.mcts import Node_mcts

class MCTS_BATCH(Brain):
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, oracle: Oracle, exploration_weight: int = 10, num_rollouts: int = 1000, time_limit: float = float("inf"), debug: bool = False) -> None:
        super().__init__()
        self.init_node = None
        self.init_board = None  # the board to be used for the next rollout
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.oracle = oracle
        self.time_limit = time_limit
        self.epsilon = 0.05  # small value to avoid time limit issues
        self.start_time = time()
        self.debug = debug

    def calculate_best_move(self, board: Board, restriction: str, value: int) -> str:
        if restriction == "depth":
            self.time_limit = float("inf")  # ignore time limit
            self.num_rollouts = value # set max rollouts
            self.run_simulation_from(board, debug=False)
            a: str = self.action_selection(training=False)
            return a 
        elif restriction == "time":
            self.time_limit = value # set time limit
            self.start_time = time() # set start time
            self.run_simulation_from(board, debug=False)
            a: str = self.action_selection(training=False)
            return a
        else:
            raise Error("Invalid restriction for MCTS")

    def choose(self, training:bool, debug:bool = False) -> Node_mcts:
        "Choose the best successor of node. (Choose a move in the game)"
        if debug:
            print("\n\nChildren of root node (sorted by visits):\n")
            for child in sorted(self.init_node.children, key=lambda x: x.N, reverse=True):
                print(f"Move: {self.init_board.stringify_move(child.move)} -> N = {child.N}, W = {child.W}, Q = {child.Q}, P = {child.P}, V = {child.V}")
        if training:
            # assert self.init_node.N == self.num_rollouts, "The number of rollouts must be equal to the number of visits to the root node."
            # print( sum([child.N for child in self.init_node.children]))
            # print(self.init_node.N)
            assert sum([child.N for child in self.init_node.children]) == self.num_rollouts
            rnd = uniform(0, 1)
            for child in self.init_node.children:
                if rnd <= (portion := child.N / self.num_rollouts):
                    return child
                rnd -= portion
            raise Error(str(rnd) + " - No child selected in MCTS.choose()")
        else:
            def score(n: Node_mcts) -> float:
                "Score function for the node. Higher is better."
                return n.N 
            # TODO: shuffle children with same best score
            return max(self.init_node.children, key=score)
        
        
    def do_rollout(self) -> None:
        "Make the tree one layer better. (Train for one iteration.)"
        leaf = self._select_and_expand()

        reward = 1 - leaf.reward()      # perché ci interessa vedere i valori di Q e W dal padre (che ha colore opposto)
        
        self._backpropagate(leaf, reward)

        return leaf.is_terminal
        

    def _select_and_expand(self) -> Node_mcts:
        "Find an unexplored descendent of `node`"
        curr_node = self.init_node
    
        curr_board = self.init_board
        
        number_of_moves = 0
    
        while True:

            if curr_node.is_unexplored or curr_node.is_terminal:
                break
            
            curr_node = self._uct_select(curr_node)  # descend a layer deeper
            
            # Aggiorno la curr_board giocando la mossa scelta
            curr_board.safe_play(curr_node.move)
            number_of_moves += 1
        

        if curr_node.is_terminal and curr_node.V == -1:
            # compute the heuristic (only once) IF DRAW because, when calling the reward function, we want to avoid the DRAW if winning
            v = self.oracle.compute_heuristic(curr_board)
            curr_node.V = v

        # expand di curr_node (non può essere terminale)
        elif curr_node.is_unexplored:
            v, pi = self.oracle.predict(curr_board)
            curr_node.expand(curr_board, v, pi)

        if number_of_moves:
            curr_board.undo(number_of_moves)

        return curr_node

    def _backpropagate(self, leaf: Node_mcts, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        while leaf is not None:
            leaf.N += 1
            leaf.W += reward
            leaf.Q = leaf.W / leaf.N
            reward = 1 - reward # TODO: invece di tenere V in [0,1], tenerlo in [-inf, inf] e fare reward = -reward
            leaf = leaf.parent
            
    def _uct_select(self, node: Node_mcts) -> Node_mcts:
        "Select a child of node, balancing exploration & exploitation"

        sqrt_N_vertex = math.sqrt(node.N)
        def uct_Norels(n:Node_mcts) -> float:
            return n.Q + self.exploration_weight * n.P * sqrt_N_vertex / (1 + n.N) #----> FIXED EXPL WEIGHT
            #return n.Q + (1 + (time() - self.start_time) / self.time_limit *(self.exploration_weight - 1)) * n.P * sqrt_N_vertex / (1 + n.N) # -----> LINEAR EXPL WEIGHT DURING TURN (NO SENSE)

        return max(node.children, key=uct_Norels)

    def run_simulation_from(self, board: Board, debug: bool=False) -> None:
        self.init_board = board
        last_move = board.moves[-1] if board.moves else None
        self.init_node = Node_mcts(last_move)

        self.init_node.set_state(board.state, board.current_player_color, board.zobrist_key, 0)
        self._select_and_expand() # expand the root node without updating N (in order to get root.N = sum(children.N))

        # if you can only do one move, return it
        if len(self.init_node.children) == 1:
            return

        terminal_states = 0

        if self.time_limit < float("inf"):
            self.num_rollouts = 0
            start_time = time()
            while time() - start_time < self.time_limit - self.epsilon:
                if self.do_rollout() and debug: # return true if new leaf is terminal
                    terminal_states += 1
                self.num_rollouts += 1
        else:
            if debug:
                for _ in tqdm(range(self.num_rollouts), desc="Rollouts", unit="rollout"):
                    if self.do_rollout() and debug: # return true if new leaf is terminal
                        terminal_states += 1
            else:
                for _ in range(self.num_rollouts):
                    if self.do_rollout() and debug: # return true if new leaf is terminal
                        terminal_states += 1
        
        if debug:
            print(f"\nTerminal states {terminal_states}/{self.num_rollouts} rollouts")

    def action_selection(self, training=False, debug:bool = False) -> str:
        # , board: Board
        # assert board == self.init_board
        node = self.choose(training=training, debug=debug)
        return self.init_board.stringify_move(node.move)
        

    def calculate_best_move(self, board: Board, restriction: str, value: int) -> str:
        self.run_simulation_from(board)
        return self.action_selection(debug=self.debug)

    def get_moves_probs(self) -> dict[Move, float]:
        #, board:Board
        # assert board == self.init_board
        
        moves_probabilities = {}
        for child in self.init_node.children:
            moves_probabilities[child.move] = child.N / self.num_rollouts

        return moves_probabilities
