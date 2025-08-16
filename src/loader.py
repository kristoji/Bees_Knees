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
    def __init__(self, folder_path, test_folder_keywords=None, max_files_per_folder=None, train_test_ratio=None):
        """
        Initialize the GraphDataset.
        
        Args:
            folder_path: String path or list of paths to folders containing data
            test_folder_keywords: List of keywords to identify test folders (default: ['test'])
            max_files_per_folder: Maximum number of files to load per subfolder (None = all)
            train_test_ratio: If provided, split data using this ratio instead of folder names
        """
        self.data = []
        self.max_files_per_folder = max_files_per_folder
        
        # Set default test folder keywords
        if test_folder_keywords is None:
            test_folder_keywords = ['test']
        self.test_folder_keywords = test_folder_keywords
        
        # Convert string path to list if necessary
        if isinstance(folder_path, str):
            folder_path = [folder_path]
        
        # Convert relative paths to absolute
        folder_path = [os.path.abspath(path) for path in folder_path]
        print(f"Processing folders: {folder_path}")
        
        # Combined train and test data
        all_data = []
        
        # Process folders
        for folder in folder_path:
            if not os.path.exists(folder):
                print(f"Warning: Folder does not exist and will be skipped: {folder}")
                continue
                
            # Load data from all folders
            cache_path = self._get_cache_path([folder], "all")
            folder_data = self._load_or_process_data([folder], cache_path, "all")
            all_data.extend(folder_data)
        
        # Split data into train and test
        self.train_data = []
        self.test_data = []
        
        # Assign data to train or test based on file paths
        for item in all_data:
            # Check if the item has a file_path attribute (we'll add this in _process_data)
            if hasattr(item, 'file_path'):
                # If any test keyword is in the file path, add to test data
                if any(keyword in item.file_path.lower() for keyword in self.test_folder_keywords):
                    self.test_data.append(item)
                else:
                    self.train_data.append(item)
            else:
                # If no file_path attribute, default to train data
                self.train_data.append(item)
        
        # Combine for backward compatibility with __getitem__ and __len__
        self.data = self.train_data + self.test_data
        
        print(f"Loaded {len(self.train_data)} training samples and {len(self.test_data)} test samples.")
    
    def _get_cache_path(self, folders, dataset_type):
        """Generate cache path based on folder list and dataset type"""
        folder_str = "_".join(sorted([os.path.basename(f) for f in folders]))
        folder_hash = hashlib.md5(folder_str.encode()).hexdigest()[:8]
        cache_filename = f"graph_cache_{dataset_type}_{folder_hash}.pkl"
        # Save cache in the first folder
        return os.path.join(folders[0], cache_filename)

    def _load_or_process_data(self, folders, cache_path, dataset_type):
        """Load data from cache or process from JSON files"""
        # Try to load from cache first
        if os.path.exists(cache_path):
            print(f"Loading {dataset_type} graphs from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded {len(data)} {dataset_type} graphs from cache.")
                return data
            except Exception as e:
                print(f"Failed to load {dataset_type} cache: {e}. Processing from scratch.")
        
        # Process from JSON files
        all_files = []
        for folder in folders:
            if not os.path.exists(folder):
                print(f"Warning: Folder does not exist and will be skipped: {folder}", flush=True)
                continue
                
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                    
                if "BAK" in subfolder:
                    print(f"Skipping subfolder: {subfolder}", flush=True)
                    continue

                print(f"Processing subfolder: {subfolder}", flush=True)
                game_dirs = glob(os.path.join(folder, subfolder, 'game_*'))

                print(f"Found {len(game_dirs)} game directories in {subfolder}", flush=True)
                
                # Limit number of directories if max_files_per_folder is set
                if self.max_files_per_folder and len(game_dirs) > 0:
                    import random
                    if len(game_dirs) > self.max_files_per_folder:
                        print(f"Limiting to {self.max_files_per_folder} game directories in {subfolder}")
                        random.shuffle(game_dirs)
                        game_dirs = game_dirs[:self.max_files_per_folder]
                
                for d in game_dirs:
                    for move_path in glob(os.path.join(d, 'move_*.json')):
                        all_files.append(move_path)
        
        print(f"Found {len(all_files)} JSON files to process for {dataset_type} dataset")
        
        # Load everything into memory
        data = []
        desc = f"Loading {dataset_type} JSON files"
        for json_path in tqdm(all_files, desc=desc):
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                processed_data = self._process_data(json_data, file_path=json_path)
                if processed_data is not None:
                    data.append(processed_data)
            except KeyboardInterrupt:
                print("Loading interrupted by user! Saving partial data...")
                break
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                continue

        print(f"Loaded {len(data)} {dataset_type} graphs from {len(all_files)} JSON files.")
        
        # Save to cache even if interrupted
        if data:
            try:
                print(f"Saving {dataset_type} graphs to cache: {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"{dataset_type.capitalize()} cache saved successfully.")
            except Exception as e:
                print(f"Failed to save {dataset_type} cache: {e}")
        
        return data

    def _process_data(self, data, file_path=None):
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
        
        graph_data = Data(x=x, edge_index=edge_index, y=v)
        
        # Store the file path for later use in train/test splitting
        if file_path:
            graph_data.file_path = file_path
        
        return graph_data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, **kwargs):
        """
        Create separate DataLoaders for train and test data.
        
        Args:
            **kwargs: Keyword arguments passed to torch_geometric.loader.DataLoader.
        
        Returns:
            tuple: (train_loader, test_loader) - DataLoader instances for train and test data.
                If no test data exists, test_loader will be None.
                If no train data exists, train_loader will be None.
        """
        train_loader = None
        test_loader = None
        
        if self.train_data:
            train_kwargs = kwargs.copy()
            train_kwargs.setdefault('shuffle', True)  # Default shuffle for training
            train_dataset = type('TrainDataset', (), {
                'data': self.train_data,
                '__getitem__': lambda self, idx: self.data[idx],
                '__len__': lambda self: len(self.data)
            })()
            train_loader = DataLoader(train_dataset, **train_kwargs)
            print(f"Created train loader with {len(self.train_data)} samples")
        else:
            print("Warning: No training data available!")
        
        if self.test_data:
            test_kwargs = kwargs.copy()
            test_kwargs['shuffle'] = False  # Never shuffle test data
            test_dataset = type('TestDataset', (), {
                'data': self.test_data,
                '__getitem__': lambda self, idx: self.data[idx],
                '__len__': lambda self: len(self.data)
            })()
            test_loader = DataLoader(test_dataset, **test_kwargs)
            print(f"Created test loader with {len(self.test_data)} samples")
        else:
            print("Warning: No test data available!")
        
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
