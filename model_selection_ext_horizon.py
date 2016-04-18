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
from model_selection import mlin_regression, summary_to_file, linear_map, rmse, mae, mape

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
        
        df['prod%ihbefore'%timeshift] = sq.fetch_production(h_hoursbefore(ts_start, timeshift),\
                                                          h_hoursbefore(ts_end, timeshift))
        for v in varnames:
            ens_mean = ens.load_ens_mean_avail_at10_series(v, ts_start, ts_end, pointcode=71699)
            ens_mean_before = ens.load_ens_mean_avail_at10_series(v,\
                                            h_hoursbefore(ts_start, timeshift),\
                                            h_hoursbefore(ts_end, timeshift),\
                                            pointcode=71699)
            df['%s%ihdiff'%(v,timeshift)] = ens_mean - ens_mean_before
    
    
    return df        
        
        
all_data = gen_fit_df(dt.datetime(2016,1,26,1), dt.datetime(2016,4,1,0), ['Tout', 'vWind', 'hum', 'sunRad'], [48,168])
y = all_data['prod']

#%%
ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,26,1), dt.datetime(2016,4,1,0))
X = all_data[['prod48hbefore', 'Tout48hdiff', 'sunRad48hdiff', 'prod168hbefore', 'Tout168hdiff', 'vWind168hdiff', 'sunRad168hdiff',  'hum168hdiff']]
est = mlin_regression(y, X, add_const=False)

fv = est.fittedvalues
plt.close('all')
sns.jointplot(fv, y)
plt.figure()
plt.plot_date(ts, y, 'b-')
plt.plot_date(ts, fv, 'r-')

err = y-fv
print rmse(err), mae(err), mape(err,y)