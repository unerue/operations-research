import numpy as np


def tminmax(di, swi, sri, hwi, hri):
    n_items = len(swi)
    # tmax
    numerator = np.sum([swi[i] for i in range(n_items)]) 
    denominator = np.sum([di[i]*hwi[i] for i in range(n_items)])
    tmax = np.sqrt((2*(sw+numerator)) / denominator)
    
    # tmin
    tmin = np.min([np.sqrt((2*swi[i]) / (di[i]*hwi[i])) for i in range(n_items)])
    return tmin, tmax