from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np
from ai.oracle import Oracle

class OracleRND(Oracle):
    def __init__(self, nn: bool = False):
        pass

    def compute_heuristic(self, board: Board) -> float:
        """
        Compute a random heuristic value for the board state in [0, 1].
        This is a placeholder implementation that returns a random value.
        """
        return np.random.rand()

    