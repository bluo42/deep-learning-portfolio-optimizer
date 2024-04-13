"""
LSTM model for portfolio optimization
Brandon Luo and Jim Skufca
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim1, input_dim2,
                 hidden_dim1, hidden_dim2,
                 output_dim1, output_dim2, num_assets,
                 num_layers1, num_layers2, dropout_prob=0.1):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim1, hidden_dim1, num_layers=num_layers1, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim2, hidden_dim2, num_layers=num_layers2, batch_first=True)
        self.fc_price = nn.Linear(hidden_dim1, output_dim1)
        self.fc_state = nn.Linear(hidden_dim2, output_dim2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_final = nn.Linear(output_dim1 + output_dim2, num_assets)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        lstm_out1, _ = self.lstm1(x1)
        lstm_out1 = self.fc_price(lstm_out1[:, -1, :])
        lstm_out1 = self.dropout(lstm_out1)
        lstm_out2, _ = self.lstm2(x2)
        lstm_out2 = self.fc_state(lstm_out2[:, -1, :])
        concat_out = torch.cat((lstm_out1, lstm_out2), dim=1)
        out = self.fc_final(concat_out)
        weights = self.softmax(out)
        return weights
