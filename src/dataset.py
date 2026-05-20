import torch
from torch.utils.data import Dataset

class MultiHorizonFI2010Dataset(Dataset):
    def __init__(self, x_data, y_data, T=100):
        """
        Args:
            x_data (Tensor): Shape [Total_Ticks, 40]
            y_data (Tensor): Shape [Total_Ticks, 5]
            T (int): Horizon window size (100)
        """
        self.x_data = x_data
        self.y_data = y_data
        self.T = T

    def __len__(self):
        return len(self.x_data) - self.T + 1

    def __getitem__(self, idx):
        x = self.x_data[idx : idx + self.T]
        y = self.y_data[idx + self.T - 1]
        return x, y.long()
