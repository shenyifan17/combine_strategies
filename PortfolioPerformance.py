#%%
import  ffn
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

#%%
def Get_Portfolio_Index_Base_Case(input_component_returns,
                        input_component_weights,
                        transaction_cost = 0,
                        weight_reference = 'portfolio',
                        initial_notional = 1000,
                        ):

    '''

    '''
    # pd.set_option('display.float.format',lambda x: '%.5f' %x)
    if not(weight_reference.lower() == 'portfolio' or  weight_reference.lower() == 'initial_notional'):
            print("Invalid weight_reference: Input either 'portfolio' or 'initial_notional' ")
            SystemExit()  

    # get parameters
    component_returns = input_component_returns.copy(deep = True)
    component_weights = input_component_weights.copy(deep = True)
    portfolio_dates = component_returns.index
    components = component_returns.columns
    input_component_weights_dates = input_component_weights.index

    # pre-allocate all to 0
    # note there is no nan
    portfolio_index = pd.Series(data=0,index=portfolio_dates)
    component_index = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    BoP_weights = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    EoP_weights = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    BoP_df = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    EoP_df = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    PnL_df_gross = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    PnL_df_net = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    dividend_value_df = pd.DataFrame(data=0,index=portfolio_dates,columns=components)
    transaction_costs_df = pd.DataFrame(data=0,index=portfolio_dates,columns=components)

    # calculate portfolio
    for date,row in component_returns.iterrows():
        print(date)

        # get previous portfolio index value
        if portfolio_dates.get_loc(date) ==0 : # first date
            reference_notional = initial_notional
        else:
            previous_period_date = portfolio_dates[portfolio_dates.get_loc(date)-1]
            reference_notional = portfolio_index.loc[previous_period_date]
        
        if weight_reference.lower() == 'initial_notional':
            reference_notional = initial_notional

        BoP_df.loc[date,:] = BoP_weights.loc[date,:] * reference_notional
        EoP_df.loc[date,:] = BoP_df.loc[date,:] * (1+(component_returns.loc[date,:]))

        if date == component_weights.index[0]:
            gross_reference_notional = reference_notional
        else:
            gross_reference_notional = EoP_df.loc[date,:].sum(skipna=True)

        PnL_df_gross.loc[date,:] = EoP_df.loc[date,:].sub(BoP_df.loc[date,:],fill_value = 0)
        tmp_EoP_override = EoP_df.loc[date,:].copy(deep = True)

        if date in input_component_weights_dates:
            # upodate weights override
            if not all(pd.isnull(component_weights.loc[date,:])):
                for component,proposed_override_weight in component_weights.loc[date,:].iteritems():
                    tmp_EoP_override.loc[component] = proposed_override_weight * gross_reference_notional

        # add transaction costs
        transaction_costs_df.loc[date,:] = abs(tmp_EoP_override.sub(EoP_df.loc[date],fill_value = 0))*transaction_cost

        # assign results
        # pnl net
        PnL_df_net.loc[date,:] = PnL_df_gross.loc[date,:].sub(transaction_costs_df.loc[date,:],fill_value = 0)
        EoP_df.loc[date,:] = tmp_EoP_override
        # portfolio index
        portfolio_index.loc[date] = reference_notional + PnL_df_net.loc[date,:].sum()

        # EoD weights
        if weight_reference.lower() == 'portfolio':
            tmp_EoP_port_index = portfolio_index.loc[date]
        elif weight_reference.lower() == 'initial_notional':
            tmp_EoP_port_index = initial_notional    
        EoP_weights.loc[date,:] = (EoP_df.loc[date,:] / tmp_EoP_port_index)

        # assign BoP next day weight
        if not (date==portfolio_dates[-1]):
            next_date = portfolio_dates[portfolio_dates.get_loc(date)+1]
            BoP_weights.loc[next_date,:] = EoP_weights.loc[date,:]

    # real return (after rebalance, transaction cost etc)
    df_component_index = (1+PnL_df_gross.sum(axis=1)/initial_notional).cumprod()

    return pd.Series([portfolio_index,BoP_weights,EoP_weights,BoP_df,EoP_df,PnL_df_gross,PnL_df_net,transaction_costs_df, df_component_index],
        index=['portfolio_index','BoP_weights','EoP_weights','BoP_df','EoP_df','PnL_df_gross','PnL_df_net','transaction_costs_df', 'df_component_index'])

def get_portfolio_stats(df_portfolio, realised_return=True, wts=None):
    """"
    Get annual_return, annual_vol, sharpe_ratio, diversification_ratio
        contribution to risk, percentage contribution to risk
        contribution to return, percentage contribution to return

        If we want to get realised risk stats, weights will be 
        dummy variable (ones) and df_portfolio input should be 
        already scaled using above function 

    
    param df_portfolio: (pandas df): fund performance index (cumulative perf of each strategy)
                                     ALREADY reconstructed to scaled performance
                                     can be absolute value or 
                                     normalised value 
                                    
          realised_return (bool): whether use realised return (1+pnl_df_net).cumprod()
                                          (in this case, wts are 1s)
                                  or using the real wts 
    """
    if realised_return:
        wts = np.ones(len(df_portfolio.columns))
    else: 
        if wts is None:
            raise ValueError('must have wts as input')

    df_component_return = df_portfolio.pct_change(1).fillna(0)
    portfolio_cov_ann = df_component_return.cov() * 252
    portfolio_sd_ann = np.sqrt(np.dot(wts.T, np.dot(portfolio_cov_ann, wts)))

    portfolio_return = np.sum(df_component_return.mean() * wts) * 252
    portfolio_sharpe = portfolio_return / portfolio_sd_ann
    diversification_ratio = np.sum(np.std(df_component_return) * np.sqrt(252) * wts) / portfolio_sd_ann

    marginal_risk_contribution = np.dot(wts.T, portfolio_cov_ann) / portfolio_sd_ann
    component_risk_contribution = marginal_risk_contribution * wts
    component_risk_contribution_pct = component_risk_contribution / portfolio_sd_ann

    component_total_return_contribution = (df_portfolio.iloc[-1] - df_portfolio.iloc[0]) / (df_portfolio.iloc[0])
    # cum_return = component_total_return_contribution.sum()
    # ann_portfolio_return = (1+cum_return) ** (252/len(df_portfolio)) - 1

    return {'volatility': portfolio_sd_ann,
            'return': portfolio_return, 
            'sharpe': portfolio_sharpe,
            'diversification_ratio': diversification_ratio,
            'marginal_risk_contribution': marginal_risk_contribution,
            'component_risk_contribution': component_risk_contribution,
            'component_risk_contribution_pct': component_risk_contribution_pct,
            'component_total_return_contribution': component_total_return_contribution}


