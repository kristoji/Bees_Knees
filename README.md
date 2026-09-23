# Bees_Knees

## Description
[UHP](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol)-compliant [Hive](https://en.wikipedia.org/wiki/Hive_(game)) in Python, inspired by [CrystalSpider](https://github.com/Crystal-Spider/hivemind) and [jonthysell](https://github.com/jonthysell/Mzinga/tree/main).

This branch (`mcts-gat`) keeps only the **MCTS + graph neural network** track: a Monte Carlo Tree Search
driven by an `Oracle` backed by a PyTorch Geometric network (GAT / GIN / GCN, selectable via `conv_type`).
The LLM fine-tuning track lives on `main`.

## Layout
```
src/engine/       Hive rules: board, moves, Zobrist hashing (no ML dependencies)
src/engineer.py   UHP command loop, entry point for the executable
src/ai/           Oracle (random / GNN), graph_network.py (GAT/GIN/GCN), MCTS and batched MCTS
src/train_gnn.py  Graph network pretraining
src/test_mcts.py  MCTS vs MCTS_BATCH benchmark
src/test/         Duel and self-play training harness (not unit tests)
src/gen_dataset/  Pro matches -> PGN -> PyG graph dataset
```

## Install the requirements
```
pip install -r requirements.txt
```
`torch` and the PyG extensions (`torch_scatter`, `torch_sparse`, ...) must be installed from their own
index matching your CUDA version; see the comments at the top of `requirements.txt`.

## Use the Makefile
```
make train-gnn    # pretrain the graph network
make play-gnn     # pretrain, then play a game against the trained network
make bench-mcts   # compare MCTS and the batched MCTS_BATCH
make selfplay     # AlphaZero-style self-play loop
make engine       # build the UHP executable
```

## Play from MzingaViewer
Build the engine executable:
```
pyinstaller ./src/engineer.py --name BeesKneesEngine --noconsole --onefile
```
Then use it from the terminal or inside the [MzingaViewer](https://github.com/jonthysell/Mzinga/releases/tag/v0.15.1).
