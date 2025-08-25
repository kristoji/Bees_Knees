GNN_PATH = models\\gnn\\pretrain_GIN_3.pt
GAME_DIR = data\\pro_matches\\board_data_tournaments
DATA_DIR = data\\LLM_dataset
SRC_DIR = src

# Default target

venv:
	.\\.venv\\Scripts\\activate
data:
	python $(SRC_DIR)\\A_LLM_data_generation.py --gnn_model $(GNN_PATH) --game_dir $(GAME_DIR) --output_dir $(DATA_DIR) --store_legal_move_embeddings
	python $(SRC_DIR)\\A_LLM_data_add_centroids.py --caches $(DATA_DIR)\\train_sequential_cache.pkl $(DATA_DIR)\\validation_sequential_cache.pkl --types board move --augment-caches --fit-on-subset --subset-fraction 0.3 --subset-max 7000


train:
	sl


.PHONY: data train
