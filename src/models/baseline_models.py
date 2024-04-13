'''
Utilities to generate weights for comparison models
Brandon Luo and Jim Skufca
'''
import torch
import pandas as pd

def generate_baseline_weights(n_rows, n_cols, device):
    weights = [0.60, 0.40] + (n_cols-2) * [0]
    baseline_weights = torch.tensor(weights * n_rows).reshape(n_rows, len(weights)).to(device)
    return baseline_weights

def generate_90_10(n_rows, n_cols, device):
    weights = [0.90, 0.10] + (n_cols-2) * [0]
    baseline_weights = torch.tensor(weights * n_rows).reshape(n_rows, len(weights)).to(device)
    return baseline_weights

def generate_20_80(n_rows, n_cols, device):
    weights = [0.2, 0.8] + (n_cols-2) * [0]
    baseline_weights = torch.tensor(weights * n_rows).reshape(n_rows, len(weights)).to(device)
    return baseline_weights

def generate_baseline_weights_df(n_rows, n_cols):
    weights = [0.60, 0.40] + (n_cols-2) * [0]
    baseline_weights = pd.DataFrame([weights] * n_rows, index=test_weights_df.index, columns=test_weights_df.columns)
    return baseline_weights

def generate_20_80_df(n_rows, n_cols):
    weights = [0.2, 0.8] + (n_cols-2) * [0]
    baseline_weights = pd.DataFrame([weights] * n_rows, index=test_weights_df.index, columns=test_weights_df.columns)
    return baseline_weights

def generate_equal_weight_df(n_rows, n_cols):
    weights = [1/n_cols] * n_cols
    baseline_weights = pd.DataFrame([weights] * n_rows, index=test_weights_df.index, columns=test_weights_df.columns)
    return baseline_weights