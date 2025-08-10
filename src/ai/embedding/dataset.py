import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """Dataset for loading graph embeddings from CSV file"""
    def __init__(self, csv_path, use_labels=False):
        self.df = pd.read_csv(csv_path)
        
        # Extract features (embeddings) and labels
        if 'label' in self.df.columns and use_labels:
            self.labels = self.df['label'].values
            self.embeddings = self.df.drop(columns=['label']).values
            self.has_labels = True
        else:
            self.embeddings = self.df.filter(regex='^emb_').values
            self.has_labels = False
            self.labels = None
        
        # Convert to float32 for PyTorch efficiency
        self.embeddings = self.embeddings.astype(np.float32)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.has_labels:
            return self.embeddings[idx], self.labels[idx]
        else:
            return self.embeddings[idx]
    
    @property
    def input_dim(self):
        return self.embeddings.shape[1]
