import pandas as pd 
import numpy as np
from scipy.optimize import minimize

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
    return [ret, np.sqrt(PortVariance), sharpe_ratio, div]

def neg_sharpe(weights, df_return):
    return get_stats(weights, df_return)[2] * (-1)

def neg_diversification(weights, df_return):
    return get_stats(weights, df_return)[3] * (-1)

def check_sum(weights):
    return np.sum(weights) - 1


def portfolio_optimisation(df_return=df_return, 
                           cutoff_date='2000-01-01', 
                           upper_bound=1,
                           drop_strategy=['equity_momentum'],
                           target='sharpe'):
    
    df_return = df_return[df_return['Date']>cutoff_date]
    
    try:
        df_return = df_return.drop(drop_strategy, axis=1)
    except:
        pass 

    
    num_strategies = len(df_return.columns) - 1
    
    init_guess = np.array(np.random.random(num_strategies))
    init_guess /= np.sum(init_guess)
    
    cons = ({'type': 'eq', 'fun': check_sum})
    bounds = tuple([(0,upper_bound) for i in range(num_strategies)])
    opt_results = minimize(fun=neg_sharpe, 
                           x0=init_guess, 
                           args=(df_return),
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=cons)

    ret, PortVariance, sharpe_ratio, div = get_stats(opt_results.x, df_return)
    
    df_portfolio = df_return[df_return.columns[1:]] * opt_results.x
    df_portfolio_sum = pd.DataFrame()
    df_portfolio_sum['Date'] = df_return['Date']
    df_portfolio_sum['perf'] = (1+df_portfolio.sum(axis=1)).cumprod()
    plt.plot(df_portfolio_sum['Date'], df_portfolio_sum['perf'])
    
    return -opt_results.fun, df_portfolio_sum, opt_results.x, ret, PortVariance, div