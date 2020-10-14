import pandas as pd 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def cal_return(df):
    """
    Change index time series into normalised return series and daily percentage return
    :param df: (pandas df) contains raw index time series, and a date column
    :return df_norm: (pandas df) normalised return, starting from 1
            df_return: (pandas df) percentage daily return
    """
    df_norm = pd.DataFrame()
    df_return = pd.DataFrame()
    df_return['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df_norm['Date'] = df_return['Date']
    strategy_list = list(df.columns[1:])
    for strategy in strategy_list:
        cleaned_name = strategy.lower().replace(' ', '_') # change column names into lower case
        df_norm[cleaned_name] = df[strategy]/df[strategy].iloc[0]
        df_return[cleaned_name] = df_norm[cleaned_name].diff()/df_norm[cleaned_name]
    return df_norm, df_return


def get_stats(weights, df_return):
    """
    Given portfolio and weights, calculate:
        a. annualised return
        b. annualised volatility
        c. annualised sharpe ratio
        d. annualised diversification ratio
    :param weights: np.array or list. weights for each strategy
    :param df_return:
    :return: dict containing return, volatility, sharpe_ratio and diversification_ratio
    """

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
    """
    Get negative objective function for scipy minimize
    :param weights: (np array) weights of strategies
    :param df_return:
    :param target: (str): optimisation target, can be 'return',
                                                      'volatility',
                                                      'sharpe_ratio',
                                                      'diversification_ratio'
    """
    return get_stats(weights, df_return)[target] * (-1)

def check_sum(weights):
    """
    Checks whether weights add up to 1, used in scipy.minimize
    :param weights:
    :return:
    """
    return np.sum(weights) - 1


def portfolio_optimisation(df_return,
                           cutoff_date='2000-01-01',
                           upper_bound=1,
                           lower_bound=0,
                           input_bound=None,
                           drop_strategy=['equity_momentum'],
                           target='sharpe_ratio',
                           optimisation_method='SLSQP'):
    """
    Run portfolio optimisation using user input
    :param df_return: (pandas df): contains daily returns
    :param cutoff_date: (str or pandas datetime_object)
    :param upper_bound: (float/int) global upper bound for all strategies
    :param lower_bound: (float/int) global lower bound for all strategies
    :param input_bound: (tuples of bounds) different upper/lower bounds for strategies
    :param drop_strategy: (list) list of strategies to drop (columns to drop from df_return)
    :param target: (str) optimisation target, 'diversification', 'sharpe_ratio', 'return', 'volatility'
    :param optimisation_method: (str) SLSQP for default non-convex optimisation (sharpe, diversification_ratio)
                                      details see scipy.optimize.minimize documentation
    :return: dict containing all optimisation information
    """
    df_return = df_return[df_return['Date']>=cutoff_date]

    # Drop unwanted strategis columns
    try:
        df_return = df_return.drop(drop_strategy, axis=1)
    except ValueError:
        pass 

    num_strategies = len(df_return.columns) - 1

    # initialise weights
    init_guess = np.array(np.random.random(num_strategies))
    init_guess /= np.sum(init_guess)
    
    cons = ({'type': 'eq', 'fun': check_sum})

    if input_bound is not None:
        bounds = input_bound
    else:
        bounds = tuple([(lower_bound, upper_bound) for i in range(num_strategies)])
    opt_results = minimize(fun=negative_target,
                           x0=init_guess, 
                           args=(df_return, target),
                           method=optimisation_method,
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
    # TO DO: add another rebalanced version of perf using
    #        monthly rebalance version
    df_portfolio_sum['perf'] = (1+df_portfolio.sum(axis=1)).cumprod()
    plt.plot(df_portfolio_sum['Date'], df_portfolio_sum['perf'])

    return {'objective': -opt_results.fun,
            'df_portfolio_sum': df_portfolio_sum, # cumulative performance
            'weights': opt_results.x,
            'return': ret, # annualised return
            'volatility': volatility, # annualised vol
            'diversification_ratio': div, # annualised diversification ratio
            'df_portfolio': df_portfolio, # strategy returns times weights
            'df_return': df_return, # raw strategy returns
            'sharpe_ratio': sharpe_ratio} # annualised sharpe_ratio
