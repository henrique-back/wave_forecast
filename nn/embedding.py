import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = self.embedding(x)
        return x
