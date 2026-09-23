SRC_DIR   = src

# Pretrained GNN checkpoint used by the MCTS oracle
GNN_PATH  = models/gnn/pretrain_GIN_3.pt

# Graph dataset produced by src/gen_dataset (pro matches -> PyG graphs)
GAME_DIR  = data/pro_matches/board_data_tournaments

# Pretrain the graph network (GIN/GAT/GCN, see kwargs_network in train_gnn.py)
train-gnn:
	python $(SRC_DIR)/train_gnn.py

# Same, then play a game against the freshly trained network
play-gnn:
	python $(SRC_DIR)/train_gnn.py --play

# Benchmark the single-threaded MCTS against the batched MCTS_BATCH
bench-mcts:
	python $(SRC_DIR)/test_mcts.py

# AlphaZero-style self-play loop: generate matches -> train -> duel -> keep winner
selfplay:
	python $(SRC_DIR)/test/trainer.py

# Build the UHP engine executable (usable from MzingaViewer)
engine:
	pyinstaller ./$(SRC_DIR)/engineer.py --name BeesKneesEngine --noconsole --onefile

.PHONY: train-gnn play-gnn bench-mcts selfplay engine
