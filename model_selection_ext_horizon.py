# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:10:28 2016

@author: Magnus Dahl
"""

import ensemble_tools as ens
import sql_tools as sq
from itertools import combinations
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import datetime as dt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

def h_hoursbefore(timestamp, h):
    return timestamp + dt.timedelta(hours=-h)

def gen_fit_df(ts_start, ts_end, varnames, timeshifts, pointcode=71699):
    """ timeshifts must be integer number of hours. Posetive values only,
        dataframe contains columns with the variables minus their value
        'timeshift' hours before. """
    
    df = pd.DataFrame()
    
    df['prod'] = sq.fetch_production(ts_start, ts_end)
    for timeshift in timeshifts:
        
        df['prod%ibefore'%timeshift] = sq.fetch_production(h_hoursbefore(ts_start, timeshift),\
                                                          h_hoursbefore(ts_end, timeshift))
        for v in varnames:
            ens_mean = ens.load_ens_mean_avail_at10_series(v, ts_start, ts_end, pointcode=71699)
            ens_mean_before = ens.load_ens_mean_avail_at10_series(v,\
                                            h_hoursbefore(ts_start, timeshift),\
                                            h_hoursbefore(ts_end, timeshift),\
                                            pointcode=71699)
            df['%s%idiff'%(v,timeshift)] = ens_mean - ens_mean_before
    
    
    return df        
        
        
        