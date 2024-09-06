import torch
import torch.utils.data as data
import pandas as pd
import numpy as np

# Define your custom dataset class (if necessary)
class SpectraDataset(data.Dataset):
    def __init__(self, X_df, y_df):

        # storing dataframes as features and labels
        self.features = X_df
        self.labels = y_df

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        # Extract features and labels from the data
        features = torch.tensor(self.features.iloc[idx].values.astype(np.float32))
        label = torch.tensor(self.labels.iloc[idx].astype(np.float32))
        return features, label