# models/vae.py
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class TemporalVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, time_window: int,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.2):
        super(TemporalVAE, self).__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            time_window=time_window,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            time_window=time_window,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        使用重参数化技巧从高斯分布中采样
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var