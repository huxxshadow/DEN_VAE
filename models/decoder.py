# models/decoder.py
import torch
import torch.nn as nn
from models.transformer_encoder import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, time_window,
                 num_heads=8, num_layers=6, dropout=0.2):
        super(Decoder, self).__init__()

        self.time_window = time_window
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, hidden_dim * time_window)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                    dim_feedforward=hidden_dim * 4,
                                                    dropout=dropout, activation='gelu')  # 使用GELU激活
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 添加层归一化
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z shape: [batch_size, latent_dim]
        hidden = self.fc(z)  # [batch_size, hidden_dim * time_window]
        hidden = hidden.view(-1, self.time_window, self.hidden_dim)  # [batch_size, time_window, hidden_dim]
        hidden = self.pos_encoder(hidden)  # [batch_size, time_window, hidden_dim]
        hidden = hidden.permute(1, 0, 2)  # [time_window, batch_size, hidden_dim]
        transformer_out = self.transformer_decoder(hidden, memory=torch.zeros_like(hidden))  # [time_window, batch_size, hidden_dim]
        transformer_out = self.layer_norm(transformer_out)  # 层归一化
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch_size, time_window, hidden_dim]
        output = self.output_layer(transformer_out)  # [batch_size, time_window, output_dim]
        return output