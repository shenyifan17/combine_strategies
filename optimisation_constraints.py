import numpy as np 

def check_sum(weights):
    """
    Checks whether weights add up to 1, used in scipy.minimize
    :param weights:
    :return:
    """
    return np.sum(weights) - 1

def cons_first_risk(cons_dict, weights_dict):

    return sum(cons_dict['first_risk'].values) - sum(weights_dict['first_rick'].values)

def cons_defensive_factors(cons_dict, weights_dict):

    return sum(cons_dict['defensive_factors'].values) - sum(weights_dict['defensive_factors'].values)

