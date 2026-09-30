from torch.utils.data import Dataset

class WaveSpectralDataset(Dataset):
    def __init__(self, X, aux, y, m0_true=None):
        self.X = X
        self.aux = aux
        self.y = y
        # Physical m0 of each target step, shape (samples, lead_time), for
        # 'shape' targets only: evaluate() needs it for the wind-sea/swell
        # labels (manuscript/decisions/log/029). Not yielded by __getitem__.
        self.m0_true = m0_true

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.aux[idx], self.y[idx]
