import numpy as np

def minmax_normalize(x):
    numerator = x - np.min(x, 0)
    denominator = np.max(x, 0) - np.min(x, 0)
    return numerator / (denominator + 1e-7)
    
    #xmax, xmin = x.max(), x.min()
    #return (x-xmin) / (xmax-xmin)

def minmax_get_norm(realx, arrx):
    xmax, xmin = arrx.max(), arrx.min()
    normx = (realx - xmin) / (xmax - xmin)
    return normx

def minmax_get_denorm(normx, xmax, xmin):
    realx = float(normx[0]) * (xmax-xmin) + xmin
    return realx

