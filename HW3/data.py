import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class StockDataset(Dataset):
    def __init__(self, stocks_data, target=0, window_size=30):
        self.window_size = window_size
        all_data = []
        self.scalers = []
        for data in stocks_data:
            scaler = MinMaxScaler((0, 1))
            data = data.reshape(-1, 1)
            scaler.fit(data)
            data = scaler.transform(data)
            data = data.squeeze()
            window_data = []
            for i in range(len(data)-window_size-1):
                window_data.append(data[i:i+window_size+1])
            window_data = np.array(window_data)
            all_data.append(window_data)
            self.scalers.append(scaler)
        all_data = np.array(all_data)
        all_data = np.transpose(all_data, axes=(1, 2, 0))
        self.data = torch.from_numpy(all_data).float()
        self.target_index = target

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = {
            'Sequence': self.data[idx, :30, :],
            "Target": self.data[idx, 30, self.target_index]}
        return sample
