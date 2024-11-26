# models/transformer_encoder.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, time_window, d_model]
        x = x.permute(1, 0, 2)  # [time_window, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]  # 增加位置编码
        return x.permute(1, 0, 2)  # [batch_size, time_window, d_model]


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, time_window, num_heads=8, num_layers=6, dropout=0.2):
        super(TransformerEncoderModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                                                    dropout=dropout, activation='gelu')  # 使用GELU激活
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(hidden_dim)  # 添加层归一化

        self.fc_mu = nn.Linear(hidden_dim * time_window, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * time_window, latent_dim)

    def forward(self, x):
        # x shape: [batch_size, time_window, input_dim]
        x = self.input_fc(x)  # [batch_size, time_window, hidden_dim]
        x = self.pos_encoder(x)  # [batch_size, time_window, hidden_dim]
        transformer_out = self.transformer_encoder(x)  # [batch_size, time_window, hidden_dim]
        transformer_out = self.layer_norm(transformer_out)  # 层归一化

        # 将所有时间步的输出拼接
        hidden = transformer_out.reshape(transformer_out.size(0), -1)  # [batch_size, hidden_dim * time_window]
        mu = self.fc_mu(hidden)  # [batch_size, latent_dim]
        log_var = self.fc_var(hidden)  # [batch_size, latent_dim]
        return mu, log_var