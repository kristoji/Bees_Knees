from torch.utils.data import Dataset
import numpy as np
import os
import time
from tqdm import tqdm

from glob import glob
import json

class NpzDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        max_wins: int = 15,
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
        file = self.files[file_idx]
        if file_idx < len(self.files_wins):
            file_path = os.path.join(self.wins_path, file)
        else:
            file_path = os.path.join(self.draws_path, file)

        data = self.data_cache[file_path]

        x = data["in_mats"][item_idx]
        y_policy = data["out_mats"][item_idx]
        y_value = data["values"][item_idx]

        #print("y_policy stats: min =", np.min(y_policy), "max =", np.max(y_policy), "mean =", np.mean(y_policy))


        return x, y_policy, y_value



class GraphDataset(Dataset):
    def __init__(self, folder_path: str):
        # folder_path is "data/pro_matches/ts/graphs/"

        self.samples = []   # list of paths to json files

        game_dirs = glob(os.path.join(folder_path, 'game_*'))
        for d in game_dirs:
            for move_path in glob(os.path.join(d, 'move_*.json')):
                self.samples.append(move_path)
        # self.samples.sort()       # ???


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path = self.samples[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)
        x = torch.tensor(data['x'], dtype=torch.float)
        edge_index = torch.tensor(data['edge_index'], dtype=torch.long).t().contiguous()
        move_adj = torch.tensor(data['move_adj'], dtype=torch.float)
        pi = torch.tensor(data['pi'], dtype=torch.float)
        v = torch.tensor(data['v'], dtype=torch.float)
        return Data(x=x, edge_index=edge_index), move_adj, pi, v