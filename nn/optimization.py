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
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 4)
    embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    
    # embed_dim should be divisible by nhead
    if embed_dim % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    # Prepare data
    n = len(density)
    train_end = int(0.7 * n) # 70% train
    val_end   = int(0.85 * n) # 15% val + 15% test

    train_density, val_density, test_density = density[:train_end], density[train_end:val_end], density[val_end:]
    train_alpha1, val_alpha1, test_alpha_1 = alpha_1[:train_end], alpha_1[train_end:val_end], alpha_1[val_end:]
    train_alpha2, val_alpha2, test_alpha_2 = alpha_2[:train_end], alpha_2[train_end:val_end], alpha_2[val_end:]
    train_r1, val_r1, test_r1 = r_1[:train_end], r_1[train_end:val_end], r_1[val_end:]

    # Prepare sequences separately
    train_X = prepare_X(train_density, train_alpha1, train_alpha2, train_r1, seq_len, lead_time)
    train_y = prepare_y(train_density, seq_len, lead_time, target=target)

    val_X   = prepare_X(val_density, val_alpha1, val_alpha2, val_r1, seq_len, lead_time)
    val_y   = prepare_y(val_density, seq_len, lead_time, target=target)

    test_x  = prepare_X(test_density, test_alpha_1, test_alpha_2, test_r1, seq_len, lead_time)
    test_y  = prepare_y(test_density, seq_len, lead_time, target=target)

    # DataLoaders
    train_loader = DataLoader(WaveSpectralDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WaveSpectralDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(WaveSpectralDataset(test_x, test_y), batch_size=batch_size, shuffle=False)


    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device: {device}')

    model = WaveHeightBaselineNN(
        num_freqs=train_X.shape[2],
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
    patience = 5
    epochs_no_improve = 0
    num_epochs = 100

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, freqs)
        val_metrics = evaluate(model, val_loader, device, freqs)

        print(f"Epoch {epoch+1}/{num_epochs} - "
            f"Train RMSE: {train_metrics['RMSE']:.4f} | "
            f"Val RMSE: {val_metrics['RMSE']:.4f} | "
            f"Val MAPE: {val_metrics['MAPE']:.2f}% | "
            f"Val CC: {val_metrics['CC']:.4f}")

        val_loss = val_metrics['RMSE']  # Optimize on RMSE

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
    
    model.load_state_dict(torch.load("best_model.pth"))

    test_metrics = evaluate(model, test_loader, device, freqs)
    print(f"Final test metrics: {test_metrics}")
    return best_val_loss