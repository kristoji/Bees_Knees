GNN_PATH = models\\gnn\\pretrain_GIN_3.pt
TEST_GAME_DIR = data\\pro_matches\\board_few_games
TEST_DATA_DIR = data\\LLM_dataset_few
TEST_CLUSTER_DIR = models\\clustering_few
BIG_GAME_DIR = data\\pro_matches\\board_data_tournaments
BIG_DATA_DIR = data\\LLM_dataset
BIG_CLUSTER_DIR = models\\clustering


#GAME_DIR = $(TEST_GAME_DIR)
#DATA_DIR = $(TEST_DATA_DIR)
#CLUSTER_DIR = $(TEST_CLUSTER_DIR)
GAME_DIR = $(BIG_GAME_DIR)
DATA_DIR = $(BIG_DATA_DIR)
CLUSTER_DIR = $(BIG_CLUSTER_DIR)

SRC_DIR = src

# Default target

venv:
	.\\.venv\\Scripts\\activate
data:
	python $(SRC_DIR)\\A_LLM_data_generation.py --gnn_model $(GNN_PATH) --game_dir $(GAME_DIR) --output_dir $(DATA_DIR) --store_legal_move_embeddings

k-means:
	python $(SRC_DIR)\\A_LLM_data_add_centroids.py --caches $(DATA_DIR)\\train_sequential_cache.pkl $(DATA_DIR)\\validation_sequential_cache.pkl --types board move --augment-caches --fit-on-subset --subset-fraction 0.3 --subset-max 100000 --output-root $(CLUSTER_DIR)


train:
	python $(SRC_DIR)\\A_LLM_trainer.py --train_cache $(DATA_DIR)\\train_sequential_cache_clustered.pkl	--val_cache $(DATA_DIR)\\validation_sequential_cache_clustered.pkl --board_centroids $(CLUSTER_DIR)\\boards\\cluster_centroids_kmeans_best.pkl --move_centroids $(CLUSTER_DIR)\\moves\\cluster_centroids_kmeans_best.pkl --epochs 2 --lr 0.00001 --batch_size 1 --output_dir models\\LLM_gpt_oss --output_format json --add_descriptions --verify_tokens --data_size 0.1 

nothing:
	neofetch

.PHONY: nothing data train venv 
