import pandas as pd 
import numpy as np 

def max_drawdown(arr):

    i = np.argmax(np.maximum.acumulate(arr) - arr)
    j = np.argmax(arr[:i])

    max_dd = (arr[i] - arr[j]) / arr[i]
    return max_dd


def clean_bbg_df(df_bbg):
    """"
    Clean the multi-index dataframe from bbg 
    """
    security_name = df_bbg.columns[0][0]
    df = pd.DataFrame()
    df['Date'] = df_bbg[security_name]['PX_LAST'].index 
    df[security_name[:-6]] = df_bbg[security_name]['PX_LAST'].values
    df = df.set_index('Date')

    return df 

