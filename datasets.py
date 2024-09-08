# import libraries
import torch
import torch.utils.data as data
import numpy as np

class SpectraDataset(data.Dataset):
    """
    Custom dataset class for loading and processing spectral data.

    Args:
        X_df (pd.DataFrame): DataFrame containing the features.
        y_df (pd.DataFrame): DataFrame containing the labels.
    """

    def __init__(self, X_df, y_df, direction = 'direct'):
        """
        Initializes the dataset with the provided features and labels.
        """

        # storing dataframes as features and labels
        self.X = X_df
        self.y = y_df
        self.direction = direction

    def __len__(self):
        """
        Returns the length of the dataset (number of samples).
        """

        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a X and y pair for a given index.
        """

        if self.direction == 'direct':
            # Extract features and labels from the data
            X = torch.tensor(self.X.iloc[idx].values.astype(np.float32))
            y = torch.tensor(self.y.iloc[idx].astype(np.float32))
        else:
            # Extract features and labels from the data
            X = torch.tensor(self.X.iloc[idx].astype(np.float32))
            y = torch.tensor(self.y.iloc[idx].values.astype(np.float32))

        return X, y