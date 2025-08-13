from typing import List, Optional
from engine.board import Board
from engine.enums import GameState, PlayerColor
from engine.game import Move
from random import choice, uniform

class Node_mcts():
    
    def __init__(self, move: Optional[Move], gamestate: GameState, curr_player_color: PlayerColor, hash: int, parent: 'Node_mcts' = None):
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = -1
        self.V = -1
        self.hash = hash
        self.gamestate: GameState = gamestate
        self.curr_player_color: PlayerColor = curr_player_color

        self.move: Optional[Move] = move
        self.parent: 'Node_mcts' = parent
        self.children: List['Node_mcts'] = []

        self.is_unexplored = True

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0 and self.children[0].P >= 0

    # @property
    # def is_unexplored(self) -> bool:
    #     #return (self.gamestate == GameState.IN_PROGRESS or self.gamestate == GameState.NOT_STARTED) and self.children == []

    @property
    def is_terminal(self) -> bool:
        return self.gamestate == GameState.DRAW or self.gamestate == GameState.WHITE_WINS or self.gamestate == GameState.BLACK_WINS

    def find_children(self, board: Board) -> List['Node_mcts']:
        "All possible successors of this board state"

        if not self.children:
            valid_moves = board.get_valid_moves()
            if valid_moves:
                for move in board.get_valid_moves():    
                    board.safe_play(move)            
                    child = Node_mcts(move=move, parent=self, gamestate=board.state, curr_player_color=board.current_player_color, hash=board.zobrist_key)
                    board.undo()
                    self.children.append(child)
            else:
                child = Node_mcts(move=None, parent=self)
                self.children.append(child)
        else:
            assert sum(child.N for child in self.children) >= self.N and self.N == 0, "Children N must be greater than or equal to parent N"
            self.reset_children()
        return self.children


    def expand(self, board: Board, v: float, pi: dict[Move, float]) -> List['Node_mcts']:
        "Expand the node by adding all successors"
        if self.is_expanded:
            self.reset_children()

        else:
            self.V = v

            self.find_children(board)

            for child in self.children:
                P = pi.get(child.move, 0.0)  # Set the prior probability from the policy
                child.set_p(P)

        # self.is_unexplored = False
        return self.children
    
    def reset_children(self) -> None:
        for child in self.children:
            child.N = 0
            child.W = 0
            child.Q = 0
            child.is_unexplored = True

    
    def find_random_child(self) -> 'Node_mcts':
        "Random successor of this board state (for more efficient simulation)"
        return choice(self.find_children(Board()))

    def set_p(self, P: float) -> None:
        "Set the prior probability of this node"
        self.P = P

    def get_state(self) -> GameState:
        "Returns True if the node has no children"
        return self.gamestate

    def reward(self) -> float:
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        if self.gamestate == GameState.WHITE_WINS:
            if self.curr_player_color == PlayerColor.WHITE:
                return 1    # black suicide
            else:
                return 0    # white win
        elif self.gamestate == GameState.BLACK_WINS:
            if self.curr_player_color == PlayerColor.BLACK:
                return 1    # white suicide
            else:
                return 0    # black win
        elif self.gamestate == GameState.DRAW:
            return 1 - self.V
            # # return 0
            return 0.5
        else:
            # ritorna la stima di vittoria della rete: V
            return self.V
            
    def __hash__(self) -> int:
        "Nodes must be hashable"
        return self.hash

    def __eq__(node1: 'Node_mcts', node2: 'Node_mcts') -> bool:
        "Nodes must be comparable"
        return node1.hash == node2.hash


