'''
Custom loss functions for use in model training
Brandon Luo and Jim Skufca
'''
import numpy as np
import torch

def sharpe_ratio_loss(predicted_weights, asset_returns, risk_free_rate=0, txn_cost=0):
    '''
    predicted weights: shape (B, N) where N is the number of assets, and B is the batch size
    asset_returns: shape (B, W, N) where W is the window
    risk_free_rate: tensor (B, 1)
    '''
    daily_risk_free_rate = np.power(1 + risk_free_rate, 1/252) - 1
    if txn_cost > 0:
        weights = predicted_weights.detach().cpu().numpy()
        changes_in_weights = np.diff(weights, axis=0)
        absolute_changes = np.abs(changes_in_weights)
        txn_costs_per_change = absolute_changes * txn_cost

        txn_costs_per_day = np.concatenate(([0], np.sum(txn_costs_per_change, axis=1)))
        txn_costs_tensor = torch.tensor(txn_costs_per_day, dtype=torch.float32, device=predicted_weights.device).unsqueeze(1)
        predicted_weights = predicted_weights.unsqueeze(1)
        portfolio_returns = torch.sum(predicted_weights * asset_returns, dim=2) - txn_costs_tensor
    else:
        predicted_weights = predicted_weights.unsqueeze(1)
        portfolio_returns = torch.sum(predicted_weights * asset_returns, dim=2)

    portfolio_returns = portfolio_returns - daily_risk_free_rate
    per_batch_return = torch.sum(portfolio_returns, axis=1)
    mean_portfolio_return = torch.mean(per_batch_return)
    std_portfolio_return = torch.std(per_batch_return)

    epsilon = 1e-8
    sharpe_ratio = mean_portfolio_return / (std_portfolio_return + epsilon) *  np.sqrt(252/asset_returns.shape[1])
    return -sharpe_ratio

def sharpe_ratio(predicted_weights, asset_returns, risk_free_rate=0, txn_cost=0):
    daily_risk_free_rate = np.power(1 + risk_free_rate, 1/252) - 1
    if txn_cost > 0:
        weights = predicted_weights.detach().cpu().numpy()
        changes_in_weights = np.diff(weights, axis=0)
        absolute_changes = np.abs(changes_in_weights)
        txn_costs_per_change = absolute_changes * txn_cost

        txn_costs_per_day = np.concatenate(([0], np.sum(txn_costs_per_change, axis=1)))
        txn_costs_tensor = torch.tensor(txn_costs_per_day, dtype=torch.float32, device=predicted_weights.device).unsqueeze(1)
        predicted_weights = predicted_weights.unsqueeze(1)
        portfolio_returns = torch.sum(predicted_weights * asset_returns, dim=2) - txn_costs_tensor
    else:
        predicted_weights = predicted_weights.unsqueeze(1)
        portfolio_returns = torch.sum(predicted_weights * asset_returns, dim=2)

    portfolio_returns = portfolio_returns - daily_risk_free_rate
    per_batch_return = torch.sum(portfolio_returns, axis=1)
    mean_portfolio_return = torch.mean(per_batch_return)
    std_portfolio_return = torch.std(per_batch_return)

    epsilon = 1e-8
    sharpe_ratio = mean_portfolio_return / (std_portfolio_return + epsilon) * np.sqrt(252/asset_returns.shape[1])
    return sharpe_ratio.item()


