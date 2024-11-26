# models/encoder.py
import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoderModel

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, time_window,
                 num_heads=8, num_layers=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoderModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            time_window=time_window,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x):
        mu, log_var = self.transformer_encoder(x)  # [batch_size, latent_dim] each
        return mu, log_var