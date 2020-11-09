import pandas as pd 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from optimisation_constraints import * 

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


def get_stats(weights, df_return, realised_return=False, frequency='daily'):
    """
    Given portfolio and weights, calculate:
        a. annualised return
        b. annualised volatility
        c. annualised sharpe ratio
        d. annualised diversification ratio
    This function is used in 
    - optimisation iteration
    - calculating rolling risk attribution
    - calculating realised portfolio risk attribution

    :param weights: np.array or list. weights for each strategy
    :param df_return: percentage return for each strategy 
                      (if realised_return, then df_return must be scaled
                      i.e. df_net_return from Chai's function)
    :param frequency (str): frequency of data, default daily (252), weekly (52)
    :return: dict containing return, volatility, sharpe_ratio and diversification_ratio
    """
    if realised_return:
        weights = np.ones(len(df_return.columns))

    multiplier_dict = {'daily': 252, 'weekly': 52}
    multiplier = multiplier_dict[frequency] 
    
    weights = np.array(weights)
    ret = np.sum(df_return.mean() * weights) * multiplier
    portfolio_cov_ann = df_return.cov() * multiplier
    portfolio_var_ann = np.dot(weights.T, 
                               np.dot(portfolio_cov_ann, weights)) 
    portfolio_std_ann = np.sqrt(portfolio_var_ann)

    sharpe_ratio = ret / portfolio_std_ann 
    vol = np.std(df_return) * np.sqrt(multiplier)
    div = np.sum(vol * weights) / portfolio_std_ann

    marginal_risk_contribution = np.dot(weights.T, portfolio_cov_ann) / portfolio_std_ann
    component_risk_contribution = marginal_risk_contribution * weights 
    component_risk_contribution_pct = component_risk_contribution / portfolio_std_ann

    df_portfolio = (1+df_return).cumprod()
    component_total_return_contribution = (df_portfolio.iloc[-1] - df_portfolio.iloc[0]) / (df_portfolio.iloc[0])

    # HHI on explained variance of PCA

    return {'return': ret,
            'volatility': portfolio_std_ann,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': div,
            'marginal_risk_contribution': marginal_risk_contribution,
            'component_risk_contribution': component_risk_contribution,
            'component_risk_contribution_pct': component_risk_contribution_pct,
            'component_total_return_contribution': component_total_return_contribution}

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


def portfolio_optimisation(df_return,
                           upper_bound=1,
                           lower_bound=0,
                           input_bound=None,
                           target='sharpe_ratio',
                           optimisation_method='SLSQP',
                           product='GRM',
                           grm_bound={'first_risk': [0.1, 0.5], 'defensive_factors': [0.1, 0.5], 
                                              'trend': [0.1, 0.5], 'tail_risk': [0.1, 0.5]}):
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

    init_weights = np.array(np.random.rand(num_strategies))
    init_weights /= np.sum(init_weights)

    # initialise weights
    if product == 'ARP': # note weights here is a np array

        cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    elif product == 'GRM': # note weights here is a dict of np arrays

        if sum([bound[1] for bound in list(grm_bound.values())]) <= 1:
            raise ValueError('grm_bound upper bound sum should be larger than 1')
        elif sum([bound[0] for bound in list(grm_bound.values())]) >= 0.9:
            raise ValueError('grm_bound lower bound sum should be smaller than 0.9')

        cons = (
                {'type': 'eq', 
                'fun': lambda weights: np.sum(weights) - 1}, # portfolio sum to 1

                {'type': 'ineq', 
                'fun': lambda weights: grm_bound['first_risk'][1] - weights[0:11].sum()}, # not exceeding group constraints

                {'type': 'ineq',
                'fun': lambda weights: grm_bound['defensive_factors'][1] - weights[11:20].sum()},

                {'type': 'ineq',
                'fun': lambda weights: grm_bound['trend'][1] - weights[20:26].sum()},

                {'type': 'ineq',
                'fun': lambda weights: grm_bound['tail_risk'][1] - weights[26:30].sum()}, 

                {'type': 'ineq',
                'fun': lambda weights: weights[0:11].sum() - grm_bound['first_risk'][0]}, # not below group constraints
                 
                {'type': 'ineq',
                'fun': lambda weights: weights[11:20].sum() - grm_bound['defensive_factors'][0]},

                {'type': 'ineq',
                'fun': lambda weights: weights[20:26].sum() - grm_bound['trend'][0]},
                
                {'type': 'ineq',
                'fun': lambda weights: weights[26:30].sum() - grm_bound['tail_risk'][0]}

                ) # to do: change to for loop


    if input_bound is not None:
        bounds = input_bound
    else:
        bounds = [[lower_bound, upper_bound] for i in range(num_strategies)]

    opt_results = minimize(fun=negative_target,
                           x0=init_weights, 
                           args=(df_return, target),
                           method=optimisation_method,
                           bounds=bounds, 
                           constraints=cons)

    stats_results = get_stats(opt_results.x, df_return)

    ret = stats_results['return']
    volatility = stats_results['volatility']
    sharpe_ratio = stats_results['sharpe_ratio']
    div = stats_results['diversification_ratio']
    weights_dict = {strategy: weight for strategy, weight in 
                    zip(df_return.columns, opt_results.x)}

    return {'objective': -opt_results.fun,
            'weights': weights_dict,
            'return': ret, # annualised return
            'volatility': volatility, # annualised vol
            'diversification_ratio': div, # annualised diversification ratio
            'sharpe_ratio': sharpe_ratio, # annualised sharpe ratio
            'marginal_risk_contribution': stats_results['marginal_risk_contribution'],
            'component_risk_contribution': stats_results['component_risk_contribution'],
            'component_risk_contribution_pct': stats_results['component_risk_contribution_pct'],
            'component_total_return_contribution': stats_results['component_total_return_contribution']} # 


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

    time_step = df_return.index.to_series().diff().min()
    # Daily data:
    if time_step == pd.Timedelta('1 day'):
        optimisation_dates_list = list(pd.bdate_range(start=start_date, 
                                       end=end_date, 
                                       freq='BM'))[::optimisation_freq]
    # Weekly data:
    elif time_step == pd.Timedelta('7 days'):
        end_of_month_bday = pd.bdate_range(start=start_date, 
                                           end=end_date, 
                                           freq='BM').to_series().dt.dayofweek
        friday_lists = []
        for day in end_of_month_bday.index:
            while (day.dayofweek) < 4:
                day -= pd.Timedelta('1 day')
            friday_lists.append(day)
        optimisation_dates_list = friday_lists[::optimisation_freq]
    else:
        raise ValueError('minimum date difference incorrect, must be 1day or 7days')

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
                                   optimisation_freq=1, # optimise every x month
                                   target='sharpe_ratio', 
                                   optimisation_method='SLSQP',
                                   input_bounds={'2009-03-31': {'cot': [0,0.5], 'fx_trend': [0,0.3]}, 
                                   '2009-04-30': {'fx_value': [0, 0.2], 'equity_quality': [0, 0.25]}}, 
                                   grm_bound={'first_risk': 0.25, 'defensive_factors': 0.25, 
                                              'trend': 0.3, 'tail_risk': 0.2},
                                   product='GRM'):
    """"
    Caculate rolling portfolio rolling optimisation result based on user input

    Input: 
    df_return (pandas df): index must be pd.datetime object
    default_lower_bound, default_upper_bound (float): between 0,1. 
        lower and upper bounds for non-specific strategies across all dates
    window_size (int): number of months used for running optimisation. 
    optimisation_freq (int): number of months running optimisation frequency 
    target (str): optimisation target, can be "sharpe_ratio", 
                                              "diversification_ratio"
                                              "return"
    optimisation_methods (str): SLSQP default, check scipy minimize doc
    input_bounds (dict of dict): user defined optimisation bounds for specific
                                 strategy and dates

    """

    ## put in optimisation results 
    optimisation_dates_list = get_optimisation_dates(df_return=df_return,
                                                     optimisation_freq=optimisation_freq)

    optimisation_bounds = generate_opt_bounds(df_return=df_return,
                                              optimisation_dates_list=optimisation_dates_list,
                                              default_lower_bound=default_lower_bound,
                                              default_upper_bound=default_upper_bound,
                                              input_bounds=input_bounds)
                
    time_step = df_return.index.to_series().diff().min()
    # Daily data:
    if time_step == pd.Timedelta('1 day'):
        slicing_list = [[str(st_date)[:10], 
                        str(pd.bdate_range(start=st_date, 
                                           periods=window_size+1, 
                                           freq='BM')[-1])[:10]] 
                        for st_date in optimisation_dates_list]\
                        [:-int((window_size/optimisation_freq))]
    # weekly data:
    elif time_step == pd.Timedelta('7 days'):
        slicing_list = []
        for st_date in optimisation_dates_list:
            end_date = pd.bdate_range(start=st_date, 
                                      periods=window_size+1,
                                      freq='BM')[-1]
            while end_date.dayofweek < 4:
                end_date -= pd.Timedelta('1 day')
            slicing_list.append([str(st_date)[:10], str(end_date)[:10]])
        slicing_list = slicing_list[:-int((window_size/optimisation_freq))]

    rolling_optimisation_result = {}
    
    for dt_range in slicing_list:
        print(f'{dt_range[0]} ---- {dt_range[1]}')

        df_opt = df_return[(df_return.index >= dt_range[0]) &
                           (df_return.index <= dt_range[1])]

        input_bound = list(optimisation_bounds[dt_range[1]].values())
        rolling_optimisation_result[dt_range[1]] = portfolio_optimisation(df_opt,
                                                                          target=target,
                                                                          input_bound=input_bound,
                                                                          optimisation_method=optimisation_method,
                                                                          product=product)
                                                        

    return rolling_optimisation_result

def Duncans_weights(df_return, rolling_optimisation_results):

    strategy_list = df_return.columns
    opt_dates_list = list(rolling_optimisation_results.keys())
    input_bounds_max_sharpe = {}
    for opt_date in opt_dates_list:
        print(opt_date)
        lower_bound = (np.array(list(rolling_optimisation_results[opt_date]['weights'].values())) - 0.02).clip(min=0)
        upper_bound = (np.array(list(rolling_optimisation_results[opt_date]['weights'].values())) + 0.02).clip(max=0.1)
        bound_list = [[low, up] for low,up in zip(lower_bound, upper_bound)]
        bounds = {strategy: bound for strategy, bound in zip(strategy_list, bound_list)}
        
        input_bounds_max_sharpe[opt_date] = bounds
    
    return input_bounds_max_sharpe

def generate_component_weights(df_return, results):
    """"
    Generate component weights for monthly rebalance (Chai's function)

    """

    df_weights_trend = pd.DataFrame(index=df_return.columns)
    for opt_date in list(results.keys()):
        df_weights_trend[opt_date] = results[opt_date]['weights'].values()
    

    start_date, end_date = df_return.index[0], df_return.index[-1]
    sr_rebalance_dates = pd.Series(pd.bdate_range(start=start_date, 
                                                    end=end_date,
                                                    freq='BM'))
    time_step = df_return.index.to_series().diff().min()

    if time_step == pd.Timedelta('7 days'):
        # get last fridays 
        def get_fri(date):
            while date.dayofweek<4:
                date -= pd.Timedelta('1 day')
            return date 
        sr_rebalance_dates = sr_rebalance_dates.apply(get_fri)

    df_rebalance_dates = pd.DataFrame(index=sr_rebalance_dates)
    df_rebalance_dates = df_rebalance_dates.rename_axis('Date')

    df_weights = df_weights_trend.T.rename_axis('Date')
    df_weights.index = pd.to_datetime(df_weights.index)
    df_component_weights = pd.merge(df_rebalance_dates, df_weights, 
                                    on='Date', how='outer').ffill().dropna()
    

    return df_component_weights, df_weights
