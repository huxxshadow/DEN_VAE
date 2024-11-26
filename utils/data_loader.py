# utils/data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class DengueDataset(Dataset):
    def __init__(self, config, augment=False):
        self.scaler = StandardScaler()
        self.data = np.load(config.DATA_PATH + "vae_input_data.npy")
        # 标准化数据
        num_samples, time_window, input_dim = self.data.shape
        self.data = self.scaler.fit_transform(self.data.reshape(-1, input_dim)).reshape(num_samples, time_window, input_dim)
        self.data = torch.FloatTensor(self.data)
        self.time_window = config.TIME_WINDOW
        self.augment = augment

        print(f"Loaded data shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.augment:
            # 添加随机噪声
            noise = torch.randn_like(sample) * 0.05
            sample += noise
            # 随机时间偏移（例如，前移或后移）
            if torch.rand(1).item() > 0.5:
                sample = torch.roll(sample, shifts=1, dims=0)
        return sample

def get_data_loaders(config):
    train_dataset = DengueDataset(config, augment=False)
    test_dataset = DengueDataset(config, augment=False)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader