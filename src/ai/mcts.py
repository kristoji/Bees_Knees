from typing import List, Optional
from engine.board import Board
from engine.enums import GameState, PlayerColor
from engine.game import Move
from random import choice, uniform

class Node_mcts():
    
    def __init__(self, move: Optional[Move] , parent: 'Node_mcts' = None):
        self.N = 0
        self.Q = 0
        self.P = 0
        self.hash = None
        self.gamestate: GameState = None
        # next player to play
        self.player_color: PlayerColor = None
        
        self.move: Move = move
        self.parent: 'Node_mcts' = parent
        self.children: List['Node_mcts'] = []

    @property
    def is_unexplored(self) -> bool:
        return (self.gamestate == GameState.IN_PROGRESS or self.gamestate == GameState.NOT_STARTED) and not self.children

    @property
    def is_terminal(self) -> bool:
        return self.gamestate == GameState.DRAW or self.gamestate == GameState.WHITE_WINS or self.gamestate == GameState.BLACK_WINS

    def find_children(self, board: Board) -> List['Node_mcts']:
        "All possible successors of this board state"

        if not self.children:
            for move in board.get_valid_moves():
                child = Node_mcts(move=move, parent=self)
                self.children.append(child)
                
        return self.children

    def expand(self, board: Board) -> List['Node_mcts']:
        "Expand the node by adding all successors"
        # TODO: chiamare la rete per calcolare i P dei figli (e settarli in set_state) e la V del nodo corrente
        
        self.find_children(board)

        for child in self.children:
            # board._zobrist_hash.value = child.hash
            # board.safe_play(child.move)
            # child.set_state(board.state, board.current_player_color, board.zobrist_key)
            # board.undo(update_hash=False)
            board.safe_play(child.move)
            child.set_state(board.state, board.current_player_color, board.zobrist_key)
            board.undo()
        return self.children
    
    def find_random_child(self) -> 'Node_mcts':
        "Random successor of this board state (for more efficient simulation)"
        return choice(self.find_children(Board()))

    def set_state(self, state: GameState, player_color: PlayerColor, hash: int) -> None:
        "Set the game state of this node"

        self.hash = hash
        self.gamestate = state
        self.player_color = player_color

    def get_state(self) -> GameState:
        "Returns True if the node has no children"
        return self.gamestate

    def reward(self) -> float:
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        if self.gamestate == GameState.WHITE_WINS:
            if self.player_color == PlayerColor.WHITE:
                return 1
            else:
                return 0
        elif self.gamestate == GameState.BLACK_WINS:
            if self.player_color == PlayerColor.BLACK:
                return 1
            else:
                return 0
        elif self.gamestate == GameState.DRAW:
            return 0.5
        else:
            # ritorna la stima di vittoria della rete: V
            # TODO: return self.V
            return uniform(0, 1)
            
    def __hash__(self) -> int:
        "Nodes must be hashable"
        return self.hash

    def __eq__(node1: 'Node_mcts', node2: 'Node_mcts') -> bool:
        "Nodes must be comparable"
        return node1.hash == node2.hash


