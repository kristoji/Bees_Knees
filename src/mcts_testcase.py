from ai.brains import MCTS
from ai.oracle import Oracle
# from ai.oracleRND import OracleRND
from ai.oracleGNN import OracleGNN
from engineer import Engine
from engine.enums import GameState
from ai.log_utils import log_header, log_subheader, log_subsubheader
from ai.oracleGNN_batch import OracleGNN_BATCH

testcases = [
    {
        "start": "Base+MLP",
        "correct_moves": ["wS1","wB1","wG1","wA1","wM","wL","wP"],
        "desc": "starting position, all moves are correct. v should be 0.5",
        "win" : False,
    },
    {
        "start" : "Base+MLP;InProgress;Black[9];wS1;bB1 wS1-;wQ \\wS1;bG1 bB1\\;wA1 /wQ;bG2 bG1-;wB1 wQ/;bQ \\bG2;wG1 \\wB1;bG2 wS1\\;wA1 -wQ;bG2 /wG1;wG1 /wA1;bG1 wB1\\;wA2 -wA1;bA1 bQ/;wA2 wB1/",
        "correct_moves" : ["bA1 -wS1"],
        "desc" : "winning move selected",
        "win" : True, 
    },
    {
        "start" : "Base+MLP;InProgress;White[31];wS1;bB1 wS1-;wQ \\wS1;bG1 bB1\\;wA1 /wQ;bG2 bG1-;wB1 wQ/;bQ \\bG2;wG1 \\wB1;bG2 wS1\\;wA1 -wQ;bG2 /wG1;wG1 /wA1;bG1 wB1\\;wA2 -wA1;bA1 bQ/;wA2 wB1/;bA1 bQ-;wG1 -wA2;bA1 bQ/;wA2 -wA1;bA1 bQ-;wB1 bG1;bS1 \\bA1;wB1 -bS1;bS1 wG1\\;wB1 bG1;bA1 bQ/;wS1 bQ\\;bA1 wS1/;wB1 bS1;bB2 /bB1;wB1 bG1;bB2 wQ\\;wS1 /bB2;bA1 bQ/;wS1 bQ\\;bA1 wS1/;wS1 /bB2;bA1 bQ\\;wA2 -bG2;bA1 bB2\\;wS1 bB1\\;bA1 wS1\\;wA2 -wA1;bB2 bB1;wA2 -bG2;bB2 wQ\\;wA2 /wA1;bA1 bQ-;wA2 -wA1;bA1 wS1\\;wA2 -bG2;bA1 /wS1;wA2 -wA1;bA1 bQ\\;wA2 -bG2;bA1 bQ-;wA2 -wA1;bA1 /wS1",
        "correct_moves" : ["wA2 -bA1", "wA2 bA1\\", "wA2 /bA1"],
        "desc" : "pinned opponent ant",
        "win" : False,
    },
    {
        "start": "Base+MLP;InProgress;Black[18];wA1;bP wA1\\;wL \\wA1;bA1 bP-;wG1 -wL;bQ bP\\;wQ \\wG1;bA2 /bP;wA2 wL/;bA1 \\wA2;wS1 wG1\\;bA2 -wQ;wS1 /bQ;bA3 bQ/;wA3 wA2\\;bA3 /wG1;wP wS1\\;bA3 wA3-;wP bQ\\;bG1 /bA2;wG2 /wS1;bG1 \\wQ;wS2 wS1\\;bG2 -bG1;wG2 bQ/;bA1 wS2\\;wM wG1\\;bG2 wQ/;wG3 wP-;bA1 bA2\\;wA2 bA3-;bG3 -bA2;wM -bG3;bL -bG1;wS2 wG2\\",
        "correct_moves": ["bL bG2\\"],
        "desc": "winning move selected",
        "win" : True, 
    },
    {
        "start": "Base+MLP;InProgress;Black[17];wA1;bP wA1\\;wL \\wA1;bA1 /bP;wQ wA1/;bQ bP\\;wA2 /wL;bA1 wQ/;wL /bP;bA2 bA1/;wA2 bQ-;wL \\wA2;wA2 \\bA2;bA3 bA2\\;wA3 wL-;bM bA3/;wG1 -wQ;bM -wG1;wM \\wA3;bA3 wA3/;wM /bQ;bG1 -bM;wP -wM;bB1 bG1\\;wP /bP;bQ bQ\\;wS1 /wP;bG1 bA1\\;wA2 \\bA3;wL /wA1;wA3 wS1\\;bP wQ\\;bQ wM\\",
        "correct_moves": ["bA3 -bA1", "bA2 -bA1"],
        "desc": "winning move selected",
        "win" : True, 
    },
    {
        "start": "Base+MLP;InProgress;Black[25];wA1;bP /wA1;wA2 \\wA1;bQ bP\\;wL wA1/;bA1 -bP;wQ wL/;bA1 \\wQ;wA2 bQ-;bA2 bA1/;wP wL\\;bA2 wP-;wA3 wA2\\;bS1 bA2-;wG1 wA3/;bM /bP;wA3 bQ\\;bM -wA1;wG1 /bP;bM -wL;wS1 wQ-;wG1 wA1\\;wA3 /bQ;bS1 wS1/;wG2 wA2-;bA3 \\bA1;wS2 wG2-;bM /bA1;wA1 bA2-;bP /bP;wM wA1\\;bA2 wQ\\;wA3 \\bA3;bA2 bQ\\;wG3 wS1\\;bP -wG1;wG1 /wG3;bG1 /bA3;wA3 /bQ;bG1 -bS1;wG1 wP\\;bG2 bS1/;wL -wP;bG2 wQ\\;wM -bP;bL /bA3;wA1 -bL;bA2 wA1\\;wA1 bQ\\",
        "correct_moves": ["bL bM\\"],
        "desc": "winning move selected",
        "win" : True,
    },
    {
        "start": "Base+MLP;InProgress;Black[24];wS1;bG1 wS1-;wP \\wS1;bA1 bG1\\;wS2 \\wP;bS1 bA1-;wQ wS2-;bQ /bS1;wQ -wP;bA2 bG1/;wM -wS2;bS2 bA2-;wL /wQ;bA3 bQ\\;wB1 \\wM;bB1 bS1\\;wA1 -wL;bA3 -wS1;wB2 wS2-;bP bA2/;wA2 wB1/;bM bP-;wA1 bM\\;bM \\bP;wL wS2/;bB2 bB1\\;wA2 bM-;bM /bA3;wA2 wL/;bQ -bB2;wA1 \\bP;bG2 /bA1;wA3 wA2-;bM bS2\\;wA2 -wM;bL /bQ;wA2 \\wL;bG3 bM\\;wA2 -bA3;bB2 bG3\\;wG1 wA3/;bA3 wG1\\;wG2 -wA1;bA3 bL\\;wA1 -bA1;bA2 bP-;wG1 -wS1",
        "correct_moves": ["play bA3 /wM"],
        "desc": "winning move selected",
        "win" : True,
    },
    {
        "start": "Base+MLP;InProgress;Black[11];wM;bG1 wM-;wQ \\wM;bP bG1\\;wA1 /wQ;bM bG1-;wA1 bP\\;bQ bM-;wB1 /wA1;bA1 bG1/;wB1 wA1;bA1 \\wQ;wA2 /wQ;bA1 wQ-;wA2 bA1/;bQ bM/;wB1 bP;bA2 bQ\\;wM bA2-;bQ bA2/;wA1 \\bQ",
        "correct_moves": ["pass"],
        "desc": "pass selected",
        "win" : False,
    },
]


exploration_weights = [
    #1,
    #5,
    10,
    #15,
    #20,
]

gnn_oracle = OracleGNN()
gnn_oracle.load("../models/pretrain_GAT_3.pt")

# gnn_batch = OracleGNN_BATCH()
# gnn_batch.load("../models/pretrain_GAT_3.pt")



oracles = [
    # Oracle(),
    #OracleRND(),
    gnn_oracle,
    # gnn_batch,
]


#                                 TESTCASE 1
# =============================================================================
#  ORACLE       |  expl_w = 5  |  expl_w = 10  |  expl_w = 15  |  expl_w = 20         
# =============================================================================
#  Oracle       |  30/30 wins  |  30/30 wins   |  ??/30 wins   |  ?/30 wins
#  OracleRND    |  18/30 wins  |  24/30 wins   |  22/30 wins   |  13/30 wins



# TIME_LIMIT = 5.0
TIME_LIMIT = float("inf")  # Set to infinity to let mcts do 1k rollouts
TEST_DIM = 1

def test_mcts():
    """
    Test the MCTS implementation with predefined game strings.
    """

    for i, testcase in enumerate(testcases[2:3]):
        
        log_subheader(f"Running test case {i + 1}/{len(testcases)}")
        
        for oracle in oracles:

            for exploration_weight in exploration_weights:

                log_header(f"Testing MCTS with {oracle.__class__.__name__} and exploration weight {exploration_weight}")
                
                right_moves = 0

                for _ in range(TEST_DIM):

                    engine = Engine()
                    engine.newgame([testcase["start"]])
                    
                    # Run MCTS with the oracle
                    mcts = MCTS(oracle=oracle, exploration_weight=exploration_weight, time_limit=TIME_LIMIT)
                    mcts.run_simulation_from(engine.board, debug=False)
                    a:str = mcts.action_selection(training=False, debug=True)

                    v, pi = oracle.predict(engine.board)
                    print(f"Predicted value: {v}")
                    print(f"Predicted move probabilities: {pi}")

                    #print(f"Action selected: {a}")
                    engine.play(a, verbose=False)

                    if a in testcase["correct_moves"] or (testcase["win"] and engine.board.state != GameState.IN_PROGRESS):
                        right_moves += 1
                        log_subsubheader(testcase["desc"]) 
                    else:
                        log_subsubheader("NOT "+testcase["desc"] + f" (selected: {a}) instead of ({testcase['correct_moves']})")

                log_subheader(f"Right moves: {right_moves}/{TEST_DIM}")

        
def main():
    """
    Main function to run the MCTS tests.
    """
    test_mcts()

if __name__ == "__main__":
    main()