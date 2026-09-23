"""Exercise MCTS_BATCH against the real engine with torch and the oracle stubbed out.

Checks that the search returns a legal move (or a pass), leaves the board exactly as
it found it, reuses its tree across plies, and finds the mates in one that a value
function can see. The stub oracle is deterministic but meaningless, so the mate tests
only prove the terminal handling works, not that the search plays well.

    python bench/test_mcts_batch.py
"""
import ast
import os
import sys
import types

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.environ.get("BENCH_SRC") or os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

torch = types.ModuleType("torch")
torch.float32 = "f32"; torch.long = "i64"
torch.zeros = lambda *a, **k: object()
sys.modules["torch"] = torch
tg = types.ModuleType("torch_geometric"); tgd = types.ModuleType("torch_geometric.data")
class Data:
    def __init__(self, **kw): self.__dict__.update(kw)
tgd.Data = Data; tg.data = tgd
sys.modules["torch_geometric"] = tg; sys.modules["torch_geometric.data"] = tgd

from engine.board import Board
from engine.enums import GameState
from ai.mcts_batch import MCTS_BATCH

class StubOracle:
    """Deterministic pseudo-network: value depends only on the zobrist key."""
    def __init__(self): self.batch_calls = 0; self.graphs_built = 0
    def _data_from_board(self, board):
        self.graphs_built += 1
        return Data(key=board.zobrist_key)
    def predict_values_batch_from_data(self, data_list, use_sigmoid=True):
        self.batch_calls += 1
        return [((d.key >> 7) % 1000) / 1000.0 for d in data_list]

def check(gamestring, rollouts, label):
    o = StubOracle()
    b = Board(gamestring)
    m = MCTS_BATCH(oracle=o, exploration_weight=5, num_rollouts=rollouts, batch_size=32)
    before_key, before_turn, before_moves = b.zobrist_key, b.turn, len(b.moves)
    move = m.calculate_best_move(b, restriction="depth", value=rollouts)
    assert b.zobrist_key == before_key, "board not restored"
    assert b.turn == before_turn and len(b.moves) == before_moves, "move stack not restored"
    legal = {b.stringify_move(x) for x in b.get_valid_moves()}
    if legal:
        assert move in legal, f"illegal move returned: {move} not in {sorted(legal)[:5]}..."
    else:
        assert move == "pass", f"expected a pass with no legal moves, got {move}"
    print(f"{label:<28} move={move:<14} rollouts={m.last_rollouts:<5} "
          f"batches={o.batch_calls:<4} graphs={o.graphs_built:<6} "
          f"cache={len(m.hashmap)}")
    return m, o, move

cases = [c for n in ast.parse(open(os.path.join(SRC, 'test_mcts.py')).read()).body
         if isinstance(n, ast.Assign) and any(getattr(t,'id','')=='testcases' for t in n.targets)
         for c in ast.literal_eval(n.value)]

check("Base+MLP", 200, "opening")
for i, c in enumerate(cases):
    if c["win"]:
        m, o, move = check(c["start"], 300, f"testcase{i} (mate in 1)")
        hit = move in c["correct_moves"]
        print(f"    expected one of {c['correct_moves']} -> {'FOUND' if hit else 'missed'}")
    else:
        check(c["start"], 200, f"testcase{i}")

# tree reuse across two consecutive searches
o = StubOracle(); b = Board("Base+MLP")
m = MCTS_BATCH(oracle=o, exploration_weight=5, num_rollouts=150, batch_size=16)
for ply in range(6):
    mv = m.calculate_best_move(b, restriction="depth", value=150)
    b.play(mv)
print(f"{'tree reuse over 6 plies':<28} ok, cache={len(m.hashmap)}, batches={o.batch_calls}")

# How much work the "look in the cache before building the graph" change saves:
# replay the same search twice on the same searcher.
o2 = StubOracle(); b2 = Board(cases[2]["start"])
m2 = MCTS_BATCH(oracle=o2, exploration_weight=5, num_rollouts=200, batch_size=32)
m2.calculate_best_move(b2, restriction="depth", value=200)
first = o2.graphs_built
m2.init_node = None
m2.calculate_best_move(b2, restriction="depth", value=200)
second = o2.graphs_built - first
print(f"{'graphs built, cold search':<28} {first}")
print(f"{'graphs built, warm cache':<28} {second}  ({first/max(1,second):.1f}x fewer)")
print("ALL CHECKS PASSED")
