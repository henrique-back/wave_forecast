import optuna
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from nn import WaveSpectralDataset, WaveHeightBaselineNN, prepare_X, prepare_y,train_one_epoch, evaluate
from utils import set_seed

def objective(trial, *, density, alpha_1, alpha_2, r_1, freqs, lead_time, target):
    set_seed(42)
    # Sample hyperparameters
    seq_len = trial.suggest_categorical('seq_len', [12, 24, 48, 96])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 4)
    embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    
    # embed_dim should be divisible by nhead
    if embed_dim % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    # Prepare data
    X = prepare_X(density, alpha_1, alpha_2, r_1, seq_len, lead_time)
    y = prepare_y(density, seq_len, lead_time, target=target)
    
    train_size = int(0.8 * len(X))
    train_X, val_X = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    train_loader = DataLoader(WaveSpectralDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WaveSpectralDataset(val_X, val_y), batch_size=batch_size, shuffle=False)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = WaveHeightBaselineNN(
        num_freqs=X.shape[2],
        freqs=freqs, 
        target=target, 
        dropout=dropout,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        embed_dim=embed_dim
        )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0
    num_epochs = 10

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, freqs)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} - "
            f"Train MSE: {train_metrics['MSE']:.4f} | "
            f"Val MSE: {val_metrics['MSE']:.4f} | "
            f"Val MAPE: {val_metrics['MAPE']:.2f}% | "
            f"Val CC: {val_metrics['CC']:.4f}")

        val_loss = val_metrics['MSE']  # Optimize on MSE

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    return best_val_loss