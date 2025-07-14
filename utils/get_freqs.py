import torch

def get_freqs(df):
    freq_cols = []
    for col in df.columns:
        try:
            float(col)
            freq_cols.append(col)
        except ValueError:
            pass

    freq_cols_sorted = sorted(freq_cols, key=lambda x: float(x))
    freqs = [float(f) for f in freq_cols_sorted]
    freqs = torch.tensor(freqs, dtype=torch.float32)
    return freqs