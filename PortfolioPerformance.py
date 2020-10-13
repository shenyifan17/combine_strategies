#%%
import  ffn
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

def Get_Portfolio_Index(component_returns,
                        component_weights,
                        is_completion_date = None,
                        weights_override_notice = 0,
                        initial_notional = 1000,
                        initial_dollar_multiplier = 100000,
                        component_signal_multiplier = None,
                        multiplier_allow_leverage = True,
                        transaction_cost = 0,
                        long_financing = 0,
                        short_financing = 0,
                        dcf = 1/360,
                        date_frequency = 'calendar_day',
                        rf = None,
                        weight_reference = 'portfolio',
                        EoP_weights_override_include_n_period_return = 0,
                        dividend_stacked = None,
                        long_dividend_tax = None,
                        component_price = None,
                        apply_mdv_constraint = False,
                        trading_mdv = None,
                        holding_mdv = None,
                        mdv_data = None,
                        mdv_assumed_notional = None,
                        long_short_pairing = None,
                        hedge_ratio = None,
                        index_true_up = None,
                        EoD_weights_true_up = None):
    ''' 
        Calculates portfolio performance 

        Paremeters:
        component_returns: a dataframe of component returns with datetime index. 
        Portfolio output will take the same date range(date frequency) as the component_returns' datetime index.

        component_weights: a dataframe of EoP component weights overrides with datetime index. 
        component_weights's datetime index can be a subset of the component_returns's datetime index.
        BoP value initially sets to 0. EoP_weight = BoP_weights * return_of_the_period.
        If weights not provided (or weights is nan), there is no override.
        On exits: explicitly state weight = 0 to include exit transaction costs, or 
        stat weight = None (Nonetype) to exclude transaction costs

        is_completion_date: same row index and column as component weights, with bolean value to indicate completion 

        weights_override_notice: 
        0:=> weights override is the EoD weights after all costs considered
        1:=> apply weights override to T-1 index value. Eqivalent to at BoP we work out the amount to trade to achieve weights override. 

        multiplier_allow_leverage: boolean. 

        initial_notional: portfolio index starting value.

        component_signal_multiplier: adjuste EoD value by a multiplier to reflect signal strength changes. 
        component_signal_multiplier is a dataframe with dateime index. 
        If multiplier is not provided or multiplier is nan, there is no change.
        If component_weights_override is applicable, updated EoP value by override first, then the EoP value is adjusted by multiplying the component_signal_multiplier.

        transaction_cost: trading costs 

        long_financing: costs to finance long positions (BoP > 0) in annualised term.

        short_financing: costs to finance short positions (BoP < 0) in annualised term.

        dcf = discount multiplier. expressed such as: 1/360

        date_frequency = 'calendar_day' or 'weekday'

        rf: a Series of risk free rate with datetime index. 
        Requires:to have the same datetime index as the component_return datetime index.
        Actual_financing_cost = rf + financing_costs

        weight_reference: Input string either: 'portfolio' or 'initial_notional'. 
        'portfolio': weights refer to a percentage of the drifting portfolio index value.
        'initial_notional': weights refer to a percentage of the constant initial notional value.

        EoP_weights_override_include_n_period_return: Input a positive number.
        0: EoP_weights = proposed_weights.
            Equivaluent to: EoP = EoP_weights_override(proposed_weights) * reference_notional.
        1: EoP_weights = proposed_weights * current_period_return. 
            Equivaluent to: EoP = EoP_weights_override(proposed_weights) * Return(t) * reference_notional
            In the case of 1, on day t, we can achieve the same number of shares as calcualted on day t-1 (observation day).
        2: EoP_weights = proposed_weights * previous_period_return * current_period_return. 
            Equivaluent to: EoP = EoP_weights_override(proposed_weights) * Return(t-1:t) * reference_notional
        and so on

        dividend_stacked: dataframe of stacked dividend information with datetime index and columns ['Component','Dividend Amount'] 
        where Dividend Amount is the dividend per share.
        if dividend_stacked data is provided, we need to also provide price data.

        component_price: same strucutre as component _returns. provide all available prices

        apply_mdv_constraint: Input boolean. if True, also need to provide trading_mdv, holding_mdv, mdv_data, mdv_assumed_notional, long_short_pair and hedge ratio(optional)

        mdv_data: same structure as weights override. If mdv_date not provided, there is no mdv constraint check.

        mdv_assumed_notional: a Series of mdv_assumed_notional in dollar amount with datetime index. 

        long_short_pairing: df with unique pairwise_id as index and two columns['long_component','short_component']. 
        if there is no pairing, leave 'short_component' value nan
        
        hedge_ratio: df with unique pairwise_id as index with 3 columns['pairwise_id','long_component','short_component']. 'long_compnent' has value of 1, 'short_component' is the hedge ratio.
        if hedge ratio is nan, it is set to 0.

        index_true_up: Series of time series to override the portoflio index.

        EoD_weights_true_up: adjuste EoD_weights if there is index_true_up. EoD_weights_true_up takes dataframe with datetime index and 2 columns ['component','weight']

    '''

    # pd.set_option('display.float.format',lambda x: '%.5f' %x)
    if not(weight_reference.lower() == 'portfolio' or  weight_reference.lower() == 'initial_notional'):
            print("Invalid weight_reference: Input either 'portfolio' or 'initial_notional' ")
            SystemExit()  

    # pre-allocate
    portfolio_dates = component_returns.index
    components = component_returns.columns

    weights_override_dates = component_weights.index
    if component_signal_multiplier is not None:
        signal_multiplier_date = component_signal_multiplier.index
    else:
        signal_multiplier_date = []

    if rf is None:
        rf = pd.Series(data=0, index= portfolio_dates)

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
    financing_costs_df = pd.DataFrame(data=0,index=portfolio_dates,columns=components)

    # calculate portfolio
    for date,row in component_returns.iterrows():
        print(date)
        if date == pd.to_datetime('20200618'):
            print("here")
        risk_free_rate = rf.loc[date]

        # get previous portfolio index value
        if portfolio_dates.get_loc(date) ==0 : # first date
            previous_period_date = portfolio_dates[0]
            BoP_portfolio_index_value = initial_notional
        else:
            previous_period_date = portfolio_dates[portfolio_dates.get_loc(date)-1]
            BoP_portfolio_index_value = portfolio_index.loc[previous_period_date]

        # get reference(notice) portfolio index
        if weights_override_notice == 0:
            reference_index = BoP_portfolio_index_value
        elif portfolio_dates.get_loc(date) < weights_override_notice: # before notice period
            reference_index = initial_notional
        else:
            reference_date = portfolio_dates[portfolio_dates.get_loc(date)-weights_override_notice]
            reference_index = portfolio_index.loc[reference_date]

        if weight_reference.lower() == 'portfolio':
            BoP_reference_notional =reference_index
            reference_notional = reference_index
        elif weight_reference.lower() == 'initial_notional':
            BoP_reference_notional = initial_notional
            reference_notional = initial_notional

        BoP_df.loc[date,:] = BoP_weights.loc[date,:] * reference_notional
        EoP_df.loc[date,:] = BoP_df.loc[date,:] * (1+(component_returns.loc[date,:]))

        PnL_df_gross.loc[date,:] = EoP_df.loc[date,:].sub(BoP_df.loc[date,:])

        if weights_override_notice == 0 and weight_reference.lower() == 'portfolio' and date > portfolio_dates[0]:
            reference_notional = EoP_df.loc[date,:].sum()

        # EoP Override
        override_return_end_date = date
        tmp_index = portfolio_dates.get_loc(date)
        if (tmp_index-EoP_weights_override_include_n_period_return + 1) >= len(portfolio_dates):
            override_return_start_date = override_return_end_date + BDay(1)
        else:   
            override_return_start_date = portfolio_dates[tmp_index-EoP_weights_override_include_n_period_return + 1]
        tmp_EoP_override = EoP_df.loc[date,:].copy(deep = True)

        if date in weights_override_dates:
            # apply MDV constraint
            if apply_mdv_constraint == True:
                if date in mdv_data.index: # if we have MDV data on the date
                    if not all(pd.isnull(mdv_data.loc[date,:])): # if data not all nan
                        for pair_id,row in long_short_pairing.iterrows():

                            # if pair_id == 218971386 and date == pd.to_datetime('20200820'):
                            #     print("here")
                        
                            # tmp_mdv_assumed_notioanl = mdv_assumed_notional.loc[date]
                            tmp_mdv_assumed_notioanl = reference_notional * initial_dollar_multiplier
                            long_component = row['long_component']
                            short_component = row['short_component']

                            if hedge_ratio is not None:
                                tmp_hedge_ratio_df = hedge_ratio.loc[date,:]
                                tmp_hedge_ratio_df.set_index('pairwise_id',inplace = True)
                                if pair_id in tmp_hedge_ratio_df.index:
                                    tmp_hedge_ratio = tmp_hedge_ratio_df.loc[pair_id ,'short_component']
                                    if pd.isna(tmp_hedge_ratio):
                                        tmp_hedge_ratio = 0
                            else:
                                tmp_hedge_ratio = 0
                            
                            if long_component in components:
                                BoP_long_dollar = tmp_mdv_assumed_notioanl * BoP_weights.loc[date,long_component]
                                propose_long_dollar = tmp_mdv_assumed_notioanl * component_weights.loc[date,long_component]
                            else:
                                BoP_long_dollar = 0 
                                propose_long_dollar = 0 
                            
                            if short_component in components:
                                BoP_short_dollar = tmp_mdv_assumed_notioanl * BoP_weights.loc[date,short_component]
                                propose_short_dollar = tmp_mdv_assumed_notioanl * component_weights.loc[date,short_component]
                            else:
                                BoP_short_dollar = 0 
                                propose_short_dollar = 0

                            if long_component in components and (not pd.isna(mdv_data.loc[date,long_component])) :
                                long_component_mdv = mdv_data.loc[date,long_component]
                            else:
                                long_component_mdv = np.inf

                            if short_component in components and (not pd.isna(mdv_data.loc[date,short_component])) :
                                short_component_mdv = mdv_data.loc[date,short_component]
                            else:
                                short_component_mdv = np.inf

                            # check long trade constraint
                            if (propose_long_dollar > 0) and (abs(np.nansum([propose_long_dollar, -BoP_long_dollar])) > long_component_mdv * trading_mdv):
                                # round down to nearest 10,000 dollars
                                propose_long_dollar = np.floor(long_component_mdv * trading_mdv/10000)*10000
                                propose_short_dollar = propose_long_dollar * tmp_hedge_ratio
                            
                            # check long holding constraint
                            if (BoP_long_dollar > 0) and (abs(propose_long_dollar) > long_component_mdv * holding_mdv): # if we already have a position
                                # round down to nearest 10,000 dollars
                                propose_long_dollar = np.floor(long_component_mdv * holding_mdv/10000)*10000
                                propose_short_dollar = propose_long_dollar * tmp_hedge_ratio

                            # check short trade constraint
                            if (propose_short_dollar < 0) and (abs(np.nansum([propose_short_dollar,-BoP_short_dollar])) > short_component_mdv * trading_mdv):
                                # round down to nearest 10,000 dollars
                                propose_short_dollar = -np.floor(short_component_mdv * trading_mdv/10000)*10000
                                propose_long_dollar = propose_short_dollar / tmp_hedge_ratio
                            
                            # check short holding constraint
                            if (BoP_short_dollar > 0) and (abs(propose_short_dollar) > short_component_mdv * holding_mdv): # if we already have a position
                                # round down to nearest 10,000 dollars
                                propose_short_dollar = -np.floor(short_component_mdv * holding_mdv/10000)*10000
                                propose_long_dollar = propose_short_dollar / tmp_hedge_ratio

                            if long_component in components:
                                component_weights.loc[date,long_component] = propose_long_dollar / tmp_mdv_assumed_notioanl
                            if short_component in components:
                                component_weights.loc[date,short_component] = propose_short_dollar / tmp_mdv_assumed_notioanl
                
            # upodate weights override
            if not all(pd.isnull(component_weights.loc[date,:])):
                for component,proposed_override_weight in component_weights.loc[date,:].iteritems():
                    if not pd.isnull(proposed_override_weight):
                        override_returns_srs = component_returns.loc[override_return_start_date:override_return_end_date,component]
                        if override_returns_srs.empty:
                            # if EoP_weights_override_include_n_period_return = 0 , we don't want to include the period return
                            # therefore the slice start is after slice end => empty 
                            override_return_multiplier = 1
                        else:
                            override_return_multiplier = override_returns_srs.add(1).cumprod()[-1]
                        
                        actual_override_weight = proposed_override_weight * override_return_multiplier
                        tmp_EoP_override.loc[component] = actual_override_weight * reference_notional

                    if (is_completion_date is not None) and (is_completion_date.loc[date,component] == True):
                        EoP_df.loc[date,component] = 0
                        tmp_EoP_override.loc[component] = 0 
            

        # Signal Multiplier
        if date in signal_multiplier_date:
            if not all(pd.isnull(component_signal_multiplier.loc[date,:])):
                tmp_total_EoP = tmp_EoP_override.sum()
                for component,multiplier in component_signal_multiplier.loc[date,:].iteritems():
                    if not pd.isnull(multiplier):   
                        tmp_EoP_override.loc[component] =  tmp_EoP_override.loc[component] * multiplier
                if multiplier_allow_leverage == False:
                    tmp_EoP_override = (tmp_EoP_override / tmp_EoP_override.sum()) * tmp_total_EoP

        # add transaction costs
        transaction_costs_df.loc[date,:] = abs(tmp_EoP_override.sub(EoP_df.loc[date]))*transaction_cost

        # number days
        if date_frequency == 'weekday':
            ndays = np.busday_count(previous_period_date.strftime('%Y-%m-%d'),date.strftime('%Y-%m-%d'),weekmask = 'Mon Tue Wed Thu Fri')
        elif date_frequency == 'calendar_day':
            ndays = np.busday_count(previous_period_date.strftime('%Y-%m-%d'),date.strftime('%Y-%m-%d'),weekmask = 'Mon Tue Wed Thu Fri Sat Sun')
        else:
            ndays = 1

        # add financing costs
        for component,BoP_value in BoP_df.loc[date,:].iteritems():
            if BoP_value >= 0:
                # long
                financing_costs_df.loc[date,component] = BoP_value*(risk_free_rate+long_financing)*dcf*ndays
            else:
                # short
                financing_costs_df.loc[date,component] = BoP_value*(risk_free_rate+short_financing)*dcf*ndays

        # add dividend 
        if dividend_stacked is not None:
            # no dividend on first day since we need previous day price to calculate number of shares
            if (date in dividend_stacked.index) and (date!=portfolio_dates[0]):
                tmp_dividen_info = dividend_stacked.loc[date.strftime("%Y%m%d"),:]
                for date,row in tmp_dividen_info.iterrows():
                    tmp_component = row['Component']
                    if tmp_component in components:
                        tmp_no_shares = BoP_reference_notional * BoP_weights.loc[date,tmp_component] / component_price.loc[previous_period_date,tmp_component]
                        if (long_dividend_tax is not None) and tmp_no_shares > 0:
                            tmp_dividend_value = tmp_no_shares * row['Dividend Amount']*(1-long_dividend_tax)
                        else:
                            tmp_dividend_value = tmp_no_shares * row['Dividend Amount']
                        dividend_value_df.loc[date,tmp_component] = tmp_dividend_value

        # pnl net
        PnL_df_net.loc[date,:] = PnL_df_gross.loc[date,:].sub(transaction_costs_df.loc[date,:]).sub(financing_costs_df.loc[date,:]).add(dividend_value_df.loc[date,:])
        
        EoP_df.loc[date,:] = tmp_EoP_override
        # portfolio index
        portfolio_index.loc[date] = BoP_portfolio_index_value + PnL_df_net.loc[date,:].sum()

        # EoD weights
        if weight_reference.lower() == 'portfolio':
            reference_notional = portfolio_index.loc[date]
        elif weight_reference.lower() == 'initial_notional':
            reference_notional = initial_notional    
        EoP_weights.loc[date,:] = (EoP_df.loc[date,:] / reference_notional)

        # index true up
        if (index_true_up is not None) and (date in index_true_up.index):
            portfolio_index.loc[date] = index_true_up.loc[date]
            if (EoD_weights_true_up is not None) and (date in EoD_weights_true_up.index):
                EoD_tmp_srs = EoD_weights_true_up.loc[date,:].set_index('component').squeeze()  
                EoP_weights.loc[date,:]= np.nan
                EoP_weights.loc[date,EoD_tmp_srs.index] = EoD_tmp_srs

        # assign BoP next day weight if today is not the last day
        if not (date==portfolio_dates[-1]):
            next_date = portfolio_dates[portfolio_dates.get_loc(date)+1]
            BoP_weights.loc[next_date,:] = EoP_weights.loc[date,:]

    return pd.Series([portfolio_index,BoP_weights,EoP_weights,BoP_df,EoP_df,PnL_df_gross,PnL_df_net,transaction_costs_df,financing_costs_df],
        index=['portfolio_index','BoP_weights','EoP_weights','BoP_df','EoP_df','PnL_df_gross','PnL_df_net','transaction_costs_df','financing_costs_df'])

