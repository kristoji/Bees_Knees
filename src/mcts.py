from abc import ABC, abstractmethod
from typing import Set


class Node_mcts(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self) -> Set['Node_mcts']:
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self) -> 'Node_mcts':
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self) -> bool:
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self) -> float:
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self) -> int:
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1: 'Node_mcts', node2: 'Node_mcts') -> bool:
        "Nodes must be comparable"
        return True


