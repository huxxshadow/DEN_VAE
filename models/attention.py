# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 因为是双向，所以hidden_dim * 2

    def forward(self, lstm_output):
        """
        lstm_output: [batch_size, time_window, hidden_dim * 2]
        """
        scores = self.attention(lstm_output)  # [batch_size, time_window, 1]
        weights = F.softmax(scores, dim=1)    # [batch_size, time_window, 1]
        context = torch.sum(weights * lstm_output, dim=1)  # [batch_size, hidden_dim * 2]
        return context