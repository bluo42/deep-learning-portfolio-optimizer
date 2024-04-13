"""
LSTM model from Zhang et. al.
Zhang, Zihao, Stefan Zohren, and Stephen Roberts. "Deep learning for portfolio optimization." arXiv preprint arXiv:2005.13665 (2020).

Brandon Luo and Jim Skufca
"""

import torch.nn as nn

class Replication_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Replication_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        weights = self.softmax(out)
        return weights

