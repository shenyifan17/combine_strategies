import numpy as np 

def check_sum(weights):
    """
    Checks whether weights add up to 1, used in scipy.minimize
    :param weights:
    :return:
    """
    return np.sum(weights) - 1


def grm_constraints(cons_dict, weights_dict, strategy_group):

    return cons_dict[strategy_group] - sum(weights_dict[strategy_group].values)

def check_sum_grm(weights_dict):
    ssum = 0 
    for key in weights_dict.keys():
        ssum += weights_dict[key]
    return ssum - 1 