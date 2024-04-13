'''
Utilities to calculate statistics for model outputs
Brandon Luo and Jim Skufca
'''
import numpy as np

def stats(returns_df, all_test_weights, txn_costs=0, vol_target=0, vol_lookback=50):
    if vol_target:
        squared_returns = returns_df ** 2
        ewma_variance = squared_returns.ewm(span=50, adjust=False).mean()
        ewma_volatility = ewma_variance.apply(np.sqrt)
        weight_adjustment = vol_target/ewma_volatility
        all_test_weights = all_test_weights.multiply(weight_adjustment, axis=1)
        all_test_weights = all_test_weights.div(all_test_weights.sum(axis=1), axis=0)
    shifted_weights = all_test_weights.shift(1).fillna(0)
    transaction_costs = (all_test_weights - shifted_weights).abs() * txn_costs

    # port returns net of transaction costs and riskfree rate
    adjusted_return = returns_df * all_test_weights - transaction_costs
    portfolio_returns = adjusted_return.sum(axis=1) - returns_df.iloc[:, -1]
    annualized_return = np.mean(portfolio_returns) * 252

    # Annualize standard deviation
    annualized_std_dev = np.std(portfolio_returns) * np.sqrt(252)

    sharpe_ratio = (annualized_return) / annualized_std_dev

    # Downside deviation
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

    # Maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Percentage of positive returns
    positive_ratio = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)

    # Ratio between positive and negative returns
    avg_positive_returns = portfolio_returns[portfolio_returns > 0].mean()
    avg_negative_returns = portfolio_returns[portfolio_returns < 0].mean()
    pos_neg_ratio = avg_positive_returns / abs(avg_negative_returns)

    # Sortino Ratio
    sortino_ratio = (annualized_return) / downside_deviation


    performance_metrics = {
        'Metric': ['Annualized Return', 'Annualized Standard Deviation', 'Sharpe Ratio',
                  'Sortino Ratio', 'DD', 'MDD', '% Positive Returns', 'Ave P/Ave L'],
        'Value': [round(annualized_return,3), round(annualized_std_dev,3), round(sharpe_ratio,3),
                  round(sortino_ratio,3), round(downside_deviation,3), round(max_drawdown,3),
                  round(positive_ratio,3), round(pos_neg_ratio,3)]
    }
    return performance_metrics['Metric'], performance_metrics['Value']