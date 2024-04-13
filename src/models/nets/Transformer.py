"""
Transformer model for portfolio optimization
Brandon Luo and Jim Skufca
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeSeriesPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(TimeSeriesPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.positional_encodings = nn.Parameter(torch.zeros(max_len, d_model))
        self.register_parameter('pe', self.positional_encodings)

        nn.init.uniform_(self.positional_encodings, -0.1, 0.1)

    def forward(self, x):
        seq_len = x.size(0)
        pe = self.pe[:seq_len, :]

        # Add positional encoding
        x = x + pe.unsqueeze(1)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_dim1, input_dim2, embedding_dim, num_assets, num_heads=8, num_encoder_layers=3, dropout_prob=0.1):
        super(Transformer, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.num_assets = num_assets

        self.embedding1 = nn.Linear(input_dim1, embedding_dim)
        self.embedding2 = nn.Linear(input_dim2, embedding_dim)

        self.positional_encoding = TimeSeriesPositionalEncoding(d_model=embedding_dim*2, dropout=dropout_prob)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim*2, nhead=num_heads, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)

        self.fc_final = nn.Linear(embedding_dim*2, num_assets)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1_embedded = self.embedding1(x1)
        x2_embedded = self.embedding2(x2)
        x_combined = torch.cat((x1_embedded, x2_embedded), dim=2)

        x_combined = x_combined.permute(1, 0, 2) 
        x_positional = self.positional_encoding(x_combined)

        transformer_output = self.transformer_encoder(x_positional)

        output = transformer_output[-1, :, :]

        weights = self.fc_final(output)
        weights = self.softmax(weights)

        return weights