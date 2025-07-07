from torch.utils.data import Dataset
import numpy as np
import os
import time
from tqdm import tqdm

# class NpzDataset(Dataset):
#     def __init__(self, folder_path: str, max_wins: int = 15, max_percent_draws: float = 0.2):
#         """
#         max_percent_draws: float = 0.2
#         This parameter limits the number of draw files to a percentage of the total wins.
#         If there are 100 win files and max_percent_draws is 0.2, it will only load 20 draw files.
#         """
#         self.folder_path = folder_path
#         self.file_indices = []

#         self.data_cache = {}  # cache for npz data already loaded and converted

#         self.draws_path = os.path.join(folder_path, "draws")
#         self.wins_path = os.path.join(folder_path, "wins")
#         self.files_wins = [f for f in os.listdir(self.wins_path) if f.endswith(".npz")]
#         self.files_wins = self.files_wins[:max_wins]  # Limit the number of win files
#         self.files_draws = [f for f in os.listdir(self.draws_path) if f.endswith(".npz")]

#         # max_draws = int(len(self.files_wins) * max_percent_draws)
#         max_draws = round(len(self.files_wins) * max_percent_draws)
#         self.files_draws = self.files_draws[:max_draws]
#         self.files = self.files_wins + self.files_draws
#         self.file_indices = []

#         # Carica e converte in memoria tutti i dati una volta sola
#         # for file_idx, file in enumerate(self.files):
#         for file_idx, file in tqdm(enumerate(self.files), desc="Loading npz files", total=len(self.files)):
#             if file_idx < len(self.files_wins):
#                 file_path = os.path.join(self.wins_path, file)
#             else:
#                 file_path = os.path.join(self.draws_path, file)
            
#             npz_data = np.load(file_path)
#             # Converte tutto in float32 subito
#             in_mats = npz_data["in_mats"].astype(np.float32)
#             out_mats = npz_data["out_mats"].astype(np.float32)
#             values = npz_data["values"].astype(np.float32)

#             self.data_cache[file_path] = {
#                 "in_mats": in_mats,
#                 "out_mats": out_mats,
#                 "values": values
#             }

#             n_samples = len(in_mats)
#             self.file_indices.extend([(file_idx, i) for i in range(n_samples)])

#         print(f"Loaded {len(self.files_wins)} win files and {len(self.files_draws)} draw files.")

class NpzDataset(Dataset):
    def __init__(
        self,
        ts_folder: str,
        iteration: int,
        max_wins: int = 15,
        max_percent_draws: float = 0.2
    ):
        """
        ts_folder: path to the timestamp directory, e.g. "data/2025-07-07_12-00-00"
        iteration: which iteration subfolder to load, e.g. 0, 1, ...
        """
        # build the iteration folder path
        self.base_path = os.path.join(ts_folder, f"iteration_{iteration}")
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
