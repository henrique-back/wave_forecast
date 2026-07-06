from torch.utils.data import Dataset

class WaveSpectralDataset(Dataset):
    def __init__(self, X, aux, y):
        self.X = X
        self.aux = aux
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.aux[idx], self.y[idx]
