'''
General utilities for data manipulation
Brandon Luo and Jim Skufca
'''
import numpy as np

def create_sequences(input_data, state_data, returns_data, lookback_window1, lookback_window2, forward_window, interval):
    x1s, x2s, ys = [], [], []
    max_lookback = max(lookback_window1, lookback_window2)
    for i in range(0, len(input_data)-max_lookback-forward_window+1, interval):
        x1 = input_data[i+max_lookback-lookback_window1:(i+max_lookback)]
        x2 = state_data[i+max_lookback-lookback_window2:(i+max_lookback)]
        y = returns_data[(i+max_lookback):(i+max_lookback+forward_window)]  # need the next n2 returns
        x1s.append(x1)
        x2s.append(x2)
        ys.append(y)
    return np.array(x1s), np.array(x2s), np.array(ys)

