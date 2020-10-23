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
    df_return = pd.DataFrame()
    df_return['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    strategy_list = list(df.columns[1:])
    for strategy in strategy_list:
        cleaned_name = strategy.lower().replace(' ', '_') # change column names into lower case
        df_return[cleaned_name] = df[strategy].pct_change(1).fillna(0)
    df_return = df_return.set_index('Date')
    return df_return


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
                           upper_bound=1,
                           lower_bound=0,
                           input_bound=None,
                           target='sharpe_ratio',
                           optimisation_method='SLSQP'):
    """
    Run portfolio optimisation using user input
    :param df_return: (pandas df): contains daily returns
    :param upper_bound: (float/int) global upper bound for all strategies
    :param lower_bound: (float/int) global lower bound for all strategies
    :param input_bound: (tuples of bounds) different upper/lower bounds for strategies
    :param target: (str) optimisation target, 'diversification', 'sharpe_ratio', 'return', 'volatility'
    :param optimisation_method: (str) SLSQP for default non-convex optimisation (sharpe, diversification_ratio)
                                      details see scipy.optimize.minimize documentation
    :return: dict containing all optimisation information
    """

    num_strategies = len(df_return.columns)

    # initialise weights
    init_guess = np.array(np.random.randn(num_strategies))
    init_guess /= np.sum(init_guess)
    
    cons = ({'type': 'eq', 'fun': check_sum})

    if input_bound is not None:
        bounds = input_bound
    else:
        bounds = [[lower_bound, upper_bound] for i in range(num_strategies)]
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

    return {'objective': -opt_results.fun,
            'weights': opt_results.x,
            'return': ret, # annualised return
            'volatility': volatility, # annualised vol
            'diversification_ratio': div, # annualised diversification ratio
            'sharpe_ratio': sharpe_ratio} # annualised sharpe_ratio


def get_optimisation_dates(df_return,
                           start_date=None, 
                           end_date=None, 
                           optimisation_freq=1):
    """"
    Get months end business day, for rebalance purposes

    Input: 
    df_return (pd df): as output from cal_return. index must be pd.timestamp
    start_date (str/pd.timestamp): yyyy-mm-dd
    end_date (str/pd.timestamp): yyyy-mm-dd
    optimisation_freq (int): rebalance frequency, default = 1, 
                          rebalance every month
    
    Output:
    optimisation_dates_list (list of pd.timestamp): end of months rebalance dates
    """
    if df_return is not None:
        start_date, end_date = df_return.index[0], df_return.index[-1]
    else: 
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    optimisation_dates_list = list(pd.bdate_range(start=start_date, 
                                   end=end_date, 
                                   freq='BM'))[::optimisation_freq]
    return optimisation_dates_list


def generate_opt_bounds(
    df_return,          
    optimisation_dates_list,       
    default_lower_bound=0,                            
    default_upper_bound=1,
    input_bounds={'2009-03-31': {'cot': [0,0.5], 'fx_trend': [0,0.3]}, 
                 '2009-04-30': {'fx_value': [0, 0.2], 'equity_quality': [0, 0.25]}}
    ):
    """"
    Generate optimisation bounds for each rebalance and each strategy.
    The optimisation_bounds variable is a dict of dicts, which first layer
    keys are rebalance dates, sencond layer keys are strategy list, and
    values are list of two numbers (bounds between 0 and 1). Default using 
    all [0,1] for all strategies and rebalances 

    input:
    df_return: as per output from cal_return
    rebalance_dates_list: (list) as per output from above function
    default_lower/upper_bound: (float) between 0,1  applied for all strategies for all dates 
    input_bounds (dict of dict): see example

    output:
    optimisaiton_bounds (dict of dict): as per described above
    """
    strategy_list = list(df_return.columns)
    # generate optimisation bounds (initial)
    optimisation_bounds = {str(rebalance_date)[:10]: 
                           {strategy: [default_lower_bound, 
                                       default_upper_bound] for strategy in strategy_list} 
                           for rebalance_date in optimisation_dates_list}
    # update opsimisation_bounds with user input:
    if input_bounds is not None:
        for rebalance_date in input_bounds.keys():
            for strategy in input_bounds[rebalance_date]:
                optimisation_bounds[rebalance_date][strategy] = input_bounds[rebalance_date][strategy]
    
    return optimisation_bounds


def rolling_portfolio_optimisation(df_return, 
                                   default_lower_bound=0,
                                   default_upper_bound=1,
                                   window_size=36,  # months
                                   optimisation_freq=1, # rebalance every x month
                                   target='sharpe_ratio', 
                                   optimisation_method='SLSQP',
                                   input_bounds={'2009-03-31': {'cot': [0,0.5], 'fx_trend': [0,0.3]}, 
                                   '2009-04-30': {'fx_value': [0, 0.2], 'equity_quality': [0, 0.25]}}
                                   ):

    ## put in optimisation results 
    optimisation_dates_list = get_optimisation_dates(df_return=df_return,
                                                     optimisation_freq=optimisation_freq)
    
    optimisation_bounds = generate_opt_bounds(df_return=df_return,
                                              optimisation_dates_list=optimisation_dates_list,
                                              default_lower_bound=default_lower_bound,
                                              default_upper_bound=default_upper_bound,
                                              input_bounds=input_bounds)
                
    slicing_list = [[str(st_date)[:10], 
                     str(pd.bdate_range(start=st_date, 
                                    periods=36, 
                                    freq='BM')[-1])[:10]] 
                     for st_date in optimisation_dates_list]\
                     [:-int((window_size/optimisation_freq))]

    rolling_optimisation_result = {}
    
    for dt_range in slicing_list:
        print(f'optimisation period {dt_range[0]} ---- {dt_range[1]}')

        df_opt = df_return[(df_return.index >= dt_range[0]) &
                           (df_return.index <= dt_range[1])]
        
        input_bound = list(optimisation_bounds[dt_range[1]].values())
        rolling_optimisation_result[dt_range[1]] = portfolio_optimisation(df_opt,
                                                                          target=target,
                                                                          input_bound=input_bound,
                                                                          optimisation_method=optimisation_method)

    return rolling_optimisation_result