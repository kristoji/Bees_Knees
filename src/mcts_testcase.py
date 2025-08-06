from ai.brains import MCTS
from ai.oracle import Oracle
from ai.oracleRND import OracleRND
from engineer import Engine
from engine.enums import GameState
from ai.log_utils import log_header, log_subheader

test_gamestrings = [
    "Base+MLP;InProgress;Black[9];wS1;bB1 wS1-;wQ \\wS1;bG1 bB1\\;wA1 /wQ;bG2 bG1-;wB1 wQ/;bQ \\bG2;wG1 \\wB1;bG2 wS1\\;wA1 -wQ;bG2 /wG1;wG1 /wA1;bG1 wB1\\;wA2 -wA1;bA1 bQ/;wA2 wB1/",
]

exploration_weights = [
    #0.5,
    8,
    #5,
    #10,
    #15,
    #20,
]

oracles = [
    Oracle(),
    OracleRND(), 
]

# =============================================================================
#  ORACLE       |  expl_w = 5  |  expl_w = 10  |  expl_w = 15  |  expl_w = 20         
# =============================================================================
#  Oracle       |  30/30 wins  |  30/30 wins   |  ??/30 wins   |  ?/30 wins
#  OracleRND    |  18/30 wins  |  24/30 wins   |  22/30 wins   |  13/30 wins



TIME_LIMIT = 5.0
TEST_DIM = 1

def test_mcts():
    """
    Test the MCTS implementation with predefined game strings.
    """
    engine = Engine()

    for i, starting_gamestring in enumerate(test_gamestrings):
        
        log_subheader(f"Running test case {i + 1}/{len(test_gamestrings)}")
        
        engine.newgame([starting_gamestring])
        
        for oracle in oracles:

            for exploration_weight in exploration_weights:

                log_header(f"Testing MCTS with {oracle.__class__.__name__} and exploration weight {exploration_weight}")
                
                right_moves = 0

                for _ in range(TEST_DIM):
                    
                    # Run MCTS with the oracle
                    mcts = MCTS(oracle=oracle, exploration_weight=exploration_weight, time_limit=TIME_LIMIT)
                    mcts.run_simulation_from(engine.board, debug=True)
                    a:str = mcts.action_selection(training=False, debug=True)
                    #print(f"Action selected: {a}")
                    engine.play(a, verbose=False)

                    if engine.board.state != GameState.IN_PROGRESS:
                        right_moves += 1

                    engine.board.undo()

                log_subheader(f"Right moves: {right_moves}/{TEST_DIM}")
                
        
def main():
    """
    Main function to run the MCTS tests.
    """
    test_mcts()

if __name__ == "__main__":
    main()