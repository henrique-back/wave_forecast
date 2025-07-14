import torch.nn as nn

class WaveHeightBaselineNN(nn.Module):
    def __init__(self, num_freqs,  target='hs', num_channels=4, lstm_hidden_size=64, lstm_layers=2):
        super(WaveHeightBaselineNN, self).__init__()
        
        self.num_freqs = num_freqs
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size
        
        # LSTM input size is num_freqs * num_channels
        self.lstm = nn.LSTM(input_size=num_freqs * num_channels,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=False)
        
        # Output layer
        output_dim = 1 if target == 'hs' else num_freqs
        self.fc = nn.Linear(lstm_hidden_size, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_freqs, num_channels)
        
        batch_size, seq_len, num_freqs, num_channels = x.shape
        
        # Flatten frequency and channel dims
        x = x.view(batch_size, seq_len, num_freqs * num_channels)
        
        # LSTM output: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Take last time step's output for prediction
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        y_pred = self.fc(last_out)     # (batch_size, 1)
        
        return y_pred.squeeze(1)       # (batch_size,)