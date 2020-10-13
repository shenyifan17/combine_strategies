import pandas as pd 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def cal_return(df):
    df_norm = pd.DataFrame()
    df_return = pd.DataFrame()
    df_return['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df_norm['Date'] = df_return['Date']
    strategy_list = list(df.columns[1:])
    for strategy in strategy_list:
        cleaned_name = strategy.lower().replace(' ', '_')
        df_norm[cleaned_name] = df[strategy]/df[strategy].iloc[0]
        df_return[cleaned_name] = df_norm[cleaned_name].diff()/df_norm[cleaned_name]
    return df_norm, df_return


def get_stats(weights, df_return):

    weights = np.array(weights)
    ret = np.sum(df_return.mean() * weights) * 252 
    PortVariance = np.dot(weights.T, np.dot(df_return.cov(), weights)) * 252 
    sharpe_ratio = ret/np.sqrt(PortVariance)
    vol = np.std(df_return) * np.sqrt(252)
    div = np.sum(vol * weights)/np.sqrt(PortVariance)

    return {'return': ret,
            'volatility': np.sqrt(PortVariance),
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': div}

def negative_target(weights, df_return, target):
    return get_stats(weights, df_return)[target] * (-1)

def check_sum(weights):
    return np.sum(weights) - 1


def portfolio_optimisation(df_return,
                           cutoff_date='2000-01-01',
                           upper_bound=1,
                           lower_bound=0,
                           drop_strategy=['equity_momentum'],
                           target='sharpe_ratio'):
    
    df_return = df_return[df_return['Date']>=cutoff_date]
    
    try:
        df_return = df_return.drop(drop_strategy, axis=1)
    except ValueError:
        pass 

    num_strategies = len(df_return.columns) - 1
    
    init_guess = np.array(np.random.random(num_strategies))
    init_guess /= np.sum(init_guess)
    
    cons = ({'type': 'eq', 'fun': check_sum})
    bounds = tuple([(lower_bound, upper_bound) for i in range(num_strategies)])
    opt_results = minimize(fun=negative_target,
                           x0=init_guess, 
                           args=(df_return, target),
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=cons)

    stats_results = get_stats(opt_results.x, df_return)

    ret = stats_results['return']
    volatility = stats_results['volatility']
    sharpe_ratio = stats_results['sharpe_ratio']
    div = stats_results['diversification_ratio']

    df_portfolio = df_return[df_return.columns[1:]] * opt_results.x
    df_portfolio_sum = pd.DataFrame()
    df_portfolio_sum['Date'] = df_return['Date']
    df_portfolio_sum['perf'] = (1+df_portfolio.sum(axis=1)).cumprod()
    plt.plot(df_portfolio_sum['Date'], df_portfolio_sum['perf'])

    return {'objective': -opt_results.fun,
            'df_portfolio_sum': df_portfolio_sum,
            'weights': opt_results.x,
            'return': ret,
            'volatility': volatility,
            'diversification_ratio': div,
            'df_portfolio': df_portfolio,
            'df_return': df_return,
            'sharpe_ratio': sharpe_ratio}
