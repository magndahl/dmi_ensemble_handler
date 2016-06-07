import numpy as np

def mae(error):
    return np.mean(np.abs(error))


def mape(error, prod):
    return mae(error/prod)


def rmse(error):
    return np.sqrt(np.mean(error**2))

