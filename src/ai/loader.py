from torch.utils.data import Dataset
import numpy as np
import os
import time
from tqdm import tqdm

from glob import glob
import json

import torch
from torch_geometric.data import Data, Batch


class GraphDataset(Dataset):
    def __init__(self, folder_path: str):
        self.data = []
        all_files = []

        for subfolder in os.listdir(folder_path):
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

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        """Create a DataLoader with proper batching for graph data"""
        from torch_geometric.loader import DataLoader
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        

class ConvDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        max_wins: int = -1,
        max_percent_draws: float = 0.2,
        max_cache_size = 30
    ):
        self.base_path = folder_path
        self.index = []
        self.file_sample_counts = {}
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = max_cache_size

        # helper to gather npz files under a win_or_draw folder
        def gather_npz(root: str, max_files:int):
            files = []
            if not os.path.isdir(root):
                return files
            num_files = 0
            for length in os.listdir(root):  # short, long, superlong
                length_dir = os.path.join(root, length)
                if not os.path.isdir(length_dir):
                    continue
                for fname in os.listdir(length_dir):
                    if fname.endswith(".npz"):
                        files.append(os.path.join(length_dir, fname))
                        num_files += 1
                        if num_files == max_files:
                            return sorted(files)
            return sorted(files)

        # collect win files
        self.files_wins = gather_npz(os.path.join(self.base_path, "wins"), max_files=max_wins)

        # collect draw files
        self.files_draws = gather_npz(os.path.join(self.base_path, "draws"), max_files=len(self.files_wins)*max_percent_draws)

        self.files = self.files_wins + self.files_draws

        for file in tqdm(self.files, total=len(self.files), desc="Scanning npz files"):
            with np.load(file, mmap_mode='r') as data:
                n_samples = data["in_mats"].shape[0]
                self.file_sample_counts[file] = n_samples
                for i in range(n_samples):
                    self.index.append((file, i))
        
        print(
            f"Scanned {len(self.files_wins)} win files "
            f"and {len(self.files_draws)} draw files "
            f"→ {len(self.index)} total samples."
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file, inner_index = self.index[idx]

        if file not in self.cache:
            self._load_file(file)
        
        data = self.cache[file]
        x = data["in_mats"][inner_index]
        y_policy = data["out_mats"][inner_index]
        y_value = data["values"][inner_index]

        return x, y_policy, y_value

    def _load_file(self, file:str):
        if len(self.cache_order) >= self.max_cache_size:
            old_file = self.cache_order.pop(0)
            del self.cache[old_file]
        
        data = np.load(file, mmap_mode='r')
        self.cache[file] = data
        self.cache_order.append(file)

class NpzDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        max_wins: int = 50,
        max_percent_draws: float = 0.2
    ):
        # build the iteration folder path
        self.base_path = folder_path

        self.data_cache = {} 
        self.file_indices = []

        # helper to gather npz files under a win_or_draw folder
        def gather_npz(root: str):
            files = []
            if not os.path.isdir(root):
                return files
            for length in os.listdir(root):  # short, long, superlong
                length_dir = os.path.join(root, length)
                if not os.path.isdir(length_dir):
                    continue
                for fname in os.listdir(length_dir):
                    if fname.endswith(".npz"):
                        files.append(os.path.join(length_dir, fname))
            return sorted(files)

        # collect win files
        wins_all = gather_npz(os.path.join(self.base_path, "wins"))
        self.files_wins = wins_all[:max_wins]

        # collect draw files
        draws_all = gather_npz(os.path.join(self.base_path, "draws"))
        max_draws = round(len(self.files_wins) * max_percent_draws)
        self.files_draws = draws_all[:max_draws]

        # final file list
        self.files = self.files_wins + self.files_draws

        # preload everything
        for file_idx, file_path in tqdm(
            enumerate(self.files),
            total=len(self.files),
            desc="Loading npz files"
        ):
            data = np.load(file_path)
            in_mats  = data["in_mats"].astype(np.float32)
            out_mats = data["out_mats"].astype(np.float32)
            values   = data["values"].astype(np.float32)

            # cache by full path
            self.data_cache[file_path] = {
                "in_mats": in_mats,
                "out_mats": out_mats,
                "values": values
            }

            # record indices: (which file, which sample in that file)
            for sample_idx in range(len(in_mats)):
                self.file_indices.append((file_idx, sample_idx))

        print(
            f"Loaded {len(self.files_wins)} win files "
            f"and {len(self.files_draws)} draw files "
            f"→ {len(self.file_indices)} total samples."
        )

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx, item_idx = self.file_indices[idx]
        file_path = self.files[file_idx]
        # if file_idx < len(self.files_wins):
        #     file_path = os.path.join(self.base_path, "wins", file_path)
        # else:
        #     file_path = os.path.join(self.base_path, "draws", file_path)

        data = self.data_cache[file_path]

        x = data["in_mats"][item_idx]
        y_policy = data["out_mats"][item_idx]
        y_value = data["values"][item_idx]

        #print("y_policy stats: min =", np.min(y_policy), "max =", np.max(y_policy), "mean =", np.mean(y_policy))


        return x, y_policy, y_value