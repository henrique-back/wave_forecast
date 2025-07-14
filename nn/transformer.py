from .positional_encoding import PositionalEncoding
from .embedding import Embedding
import torch
import torch.nn as nn
from utils import get_start_token

class WaveHeightBaselineNN(nn.Module):
    def __init__(self,
                freqs,    
                num_freqs,
                target='hs',
                num_channels=4,
                nhead=2, 
                num_encoder_layers=2,
                num_decoder_layers=2, 
                embed_dim=16, 
                batch_first=True,
                max_len=500,
                dropout=0.1):

        super().__init__()
        
        self.num_freqs = num_freqs
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.target = target

        input_dim = num_freqs * num_channels

        # Embedding layer
        self.encoder_embedding = Embedding(input_dim, embed_dim)
        self.decoder_embedding = Embedding(1 if self.target == 'hs' else num_freqs, embed_dim)

        # Positional Encoding layer
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)

        # Trasnformer
        self.transformer = nn.Transformer(nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          d_model=embed_dim, 
                                          batch_first=batch_first,
                                          )

        # Output layer — predict either 1 value (hs) or num_freqs (density)
        output_dim = 1 if self.target == 'hs' else num_freqs
        self.predictor = nn.Linear(embed_dim, output_dim)

    
    def forward(self, src, tgt):
        # src: (batch_size, src_seq_len, num_freqs, num_channels)
        # tgt: (batch_size, tgt_seq_len, 1 or num_freqs)
        batch_size, src_seq_len, num_freqs, num_channels = src.shape
        src = src.view(batch_size, src_seq_len, num_freqs * num_channels)

        # Embedding + positional encoding
        src = self.encoder_embedding(src)
        src = self.pos_encoder(src)

        tgt = self.decoder_embedding(tgt)
        tgt = self.pos_encoder(tgt)

        # Generate causal mask for decoder
        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        # Transformer forward
        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask
        )

        # Final output layer
        output = self.predictor(output)  # (batch_size, tgt_seq_len, output_dim)
        return output
    
    @torch.no_grad() #disables gradient tracking during inference (faster & avoids memory leaks).
    def infer(self, src, freqs, lead_time):
        """Autoregressive inference for multi-step forecasting.

        Args:
            src (torch.Tensor): Input to the encoder [batch_size, src_seq_len, num_freqs, num_channels]
            lead_time (int): Number of future time steps to forecast.

        Returns:
            torch.Tensor: Forecasted sequence [batch_size, lead_time, output_dim]
        """
        batch_size = src.size(0)
        output_dim = 1 if self.target == 'hs' else self.num_freqs

        # Prepare output tensor for decoder inputs + future predictions
        output = torch.zeros((batch_size, lead_time + 1, output_dim), device=src.device)

        start_token = get_start_token(src, self.target, freqs, src.device)
        output[:, 0] = start_token

        # Autoregressive decoding loop
        for i in range(lead_time):
            preds = self.forward(src, output)  # Predict full sequence
            output[:, i + 1] = preds[:, i]     # Take next step prediction

        return output[:, 1:]  # Discard initial start token
