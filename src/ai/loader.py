from torch.utils.data import Dataset
import os
from tqdm import tqdm
from glob import glob
import json
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import pickle
import hashlib
import pandas as pd
from torch.utils.data import random_split, DataLoader as TorchDataLoader

class GraphDataset(Dataset):
    def __init__(self, folder_path: str):
        self.data = []
        
        # Find existing cache file (any .pkl file in the folder)
        pkl_files = glob(os.path.join(folder_path, "*.pkl"))
        
        if pkl_files:
            self.cache_path = pkl_files[0]  # Use the first (and hopefully only) .pkl file
            print(f"Loading graphs from cache: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"Loaded {len(self.data)} graphs from cache.")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Processing from scratch.")
        
        # Create cache file path if no existing cache found
        folder_hash = hashlib.md5(folder_path.encode()).hexdigest()[:8]
        cache_filename = f"graph_cache_{folder_hash}.pkl"
        self.cache_path = os.path.join(folder_path, cache_filename)
        
        # Process from JSON files if cache doesn't exist or failed to load
        all_files = []

        for subfolder in os.listdir(folder_path):
            
            #if the folder contains "BAK" skip it, used to remove files that are not needed
            if "BAK" in subfolder:
                print(f"Skipping subfolder: {subfolder}", flush=True)
                continue

            print(f"Processing subfolder: {subfolder}", flush=True)
            game_dirs = glob(os.path.join(folder_path, subfolder, 'game_*'))

            print(f"Found {len(game_dirs)} game directories in {subfolder}", flush=True)
            for d in game_dirs:
                for move_path in glob(os.path.join(d, 'move_*.json')):
                    all_files.append(move_path)
        
        # Load everything into memory at initialization
        for json_path in tqdm(all_files, desc="Loading JSON files into memory"):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            processed_data = self._process_data(data)
            # Only add valid data (skip None/empty graphs)
            if processed_data is not None:
                self.data.append(processed_data)

        print(f"Loaded {len(self.data)} graphs from {len(all_files)} JSON files.")
        
        # Save to cache for future use
        try:
            print(f"Saving graphs to cache: {self.cache_path}")
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.data, f)
            print("Cache saved successfully.")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _process_data(self, data):
        x = data['x']
        temp = []
        for el in x:
            flattened = []
            for sub_el in el:
                if isinstance(sub_el, list):
                    flattened.extend(sub_el)
                else:
                    flattened.append(sub_el)
            temp.append(flattened)

        # Skip empty graphs
        if len(temp) == 0:
            return None

        x = torch.tensor(temp, dtype=torch.float)
        edge_index = torch.tensor(data['edge_index'], dtype=torch.long).contiguous()
        v = torch.tensor(data.get('v', 0.0), dtype=torch.float)
        v = (v + 1) / 2.0
        if v.dim() == 0:
            v = v.unsqueeze(0)
        
        return Data(x=x, edge_index=edge_index, y=v)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, train_size=None, **kwargs):
        """
        Create a DataLoader with proper batching for graph data.
        
        Args:
            train_size (float, optional): If provided, returns a tuple of (train_loader, test_loader)
                                         with the dataset split according to this ratio (e.g., 0.8 for 80% train).
            **kwargs: Keyword arguments passed directly to torch_geometric.loader.DataLoader.
                     Common arguments include:
                     - batch_size (int): Number of samples per batch (default: 1)
                     - shuffle (bool): Whether to shuffle data at every epoch (default: False)
                     - num_workers (int): Number of subprocesses for data loading (default: 0)
                     - drop_last (bool): Drop the last incomplete batch (default: False)
                     - pin_memory (bool): Pin memory for faster GPU transfer (default: False)
        
        Returns:
            If train_size is None:
                torch_geometric.loader.DataLoader: A single DataLoader instance for the entire dataset.
            If train_size is provided:
                Tuple[DataLoader, DataLoader]: A tuple of (train_loader, test_loader).
        """
        if train_size is None:
            return DataLoader(self, **kwargs)
        
        # Split dataset into train and test
        train_size = int(train_size * len(self))
        test_size = len(self) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        
        # Create train DataLoader with provided kwargs
        train_loader = DataLoader(train_dataset, **kwargs)
        
        # Create test DataLoader with same kwargs but ensure shuffle=False
        test_kwargs = {**kwargs, "shuffle": False}
        test_loader = DataLoader(test_dataset, **test_kwargs)
        
        return train_loader, test_loader
        
class EmbeddingDataset(Dataset):
    """Dataset for loading graph embeddings from CSV file"""
    def __init__(self, csv_path, use_labels=False):
        self.df = pd.read_csv(csv_path)
        
        # Extract features (embeddings) and labels
        if 'label' in self.df.columns and use_labels:
            self.labels = torch.tensor(self.df['label'].values, dtype=torch.float32)
            self.embeddings = torch.tensor(self.df.drop(columns=['label']).values, dtype=torch.float32)
            self.has_labels = True
        else:
            self.embeddings = torch.tensor(self.df.filter(regex='^emb_').values, dtype=torch.float32)
            self.has_labels = False
            self.labels = None
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.has_labels:
            return self.embeddings[idx], self.labels[idx]
        else:
            return self.embeddings[idx]

    def _preload_to_gpu(self):
        if torch.cuda.is_available():
            self.embeddings = self.embeddings.to("cuda")
            if self.has_labels:
                self.labels = self.labels.to("cuda")

    @property
    def input_dim(self):
        return self.embeddings.shape[1]

    def get_dataloader(self, train_size=0.8, **kwargs):
        """
        Create a DataLoader for the dataset.

        Args:
            **kwargs: Keyword arguments passed to the DataLoader.

        Returns:
            DataLoader: A DataLoader instance for the dataset.
        """

        train_dataset, test_dataset = random_split(self, [train_size, 1 - train_size])

        train_loader = TorchDataLoader(train_dataset, **kwargs)

        test_kwargs = {**kwargs, "shuffle": False}
        test_loader = TorchDataLoader(test_dataset, **test_kwargs)

        return train_loader, test_loader
