from torch.utils.data import Dataset
import numpy as np
import os

class NpzDataset(Dataset):
    def __init__(self, folder_path: str, max_percent_draws: float = 0.2):
        """
        max_percent_draws: float = 0.2
        This parameter limits the number of draw files to a percentage of the total wins.
        If there are 100 win files and max_percent_draws is 0.2, it will only load 20 draw files.
        """
        self.folder_path = folder_path
        self.file_indices = []


        self.draws_path = os.path.join(folder_path, "draws")
        self.wins_path = os.path.join(folder_path, "wins")
        self.files_wins = [f for f in os.listdir(self.wins_path) if f.endswith(".npz")]
        self.files_draws = [f for f in os.listdir(self.draws_path) if f.endswith(".npz")]

        max_draws = int(len(self.files_wins) * max_percent_draws)
        self.files_draws = self.files_draws[:max_draws]
        self.files = self.files_wins + self.files_draws
        self.file_indices = []
        # Precalcola gli indici interni a ogni file
        for file_idx, file in enumerate(self.files):
            file_path = os.path.join(self.folder_path, file)
            data = np.load(file_path)
            n_samples = len(data["in_mats"])
            self.file_indices.extend([(file_idx, i) for i in range(n_samples)])



    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx, item_idx = self.file_indices[idx]
        file = self.files[file_idx]
        file_path = os.path.join(self.folder_path, file)
        data = np.load(file_path)

        x = np.array(data["in_mats"][item_idx], dtype=np.float32)
        y_policy = np.array(data["out_mats"][item_idx], dtype=np.float32)
        y_value = np.array(data["values"][item_idx], dtype=np.float32)

        return x, y_policy, y_value
