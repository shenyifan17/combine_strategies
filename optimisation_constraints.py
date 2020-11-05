import numpy as np 

def check_sum(weights):
    """
    Checks whether weights add up to 1, used in scipy.minimize
    :param weights:
    :return:
    """
    return np.sum(weights) - 1

def grm_constraints(cons_dict, weights_dict):


    first_risk = cons_dict['first_risk'] - weights_dict['first_rick']
    tail_risk = cons_dict['tail_risk'] - weights_dict['tail_risk']
    trend = cons_dict['trend']  - weights_dict['trend']

    total = np.sum(wts for wts in weights_dict[group] for group in weights_dict.keys())


    return diff_prod