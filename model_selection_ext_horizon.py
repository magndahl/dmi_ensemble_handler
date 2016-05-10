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
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn import linear_model, cross_validation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_selection import gen_all_combinations, rmse, mae, mape
import pandas as pd

plt.close('all')

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

reload_data = False
if reload_data:        
    timelags = [48,168]
    all_data = gen_fit_df(dt.datetime(2016,1,26,1), dt.datetime(2016,4,1,0), ['Tout', 'vWind', 'hum', 'sunRad'], timelags)
y = all_data['prod']

#%%
ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,26,1), dt.datetime(2016,4,1,0))
X = all_data.ix[:, all_data.columns !='prod']


#%%
lr = linear_model.LinearRegression(fit_intercept=False)


predicted = cross_val_predict(lr, X, y, cv=25)
plt.figure()
plt.plot(y)
plt.plot(predicted, 'r')
sns.jointplot(pd.Series(predicted), y)
score = cross_val_score(lr, X, y, cv=25, scoring='mean_absolute_error' )

lr.fit(X,y)

var_combs = gen_all_combinations(X.columns)
maes = np.zeros(len(var_combs))
rmses = np.zeros(len(var_combs))
for v, i in zip(var_combs, range(len(var_combs))):
    predicted = cross_val_predict(lr, all_data[v], y, cv=4)
    maes[i] = mae(predicted-y)
    rmses[i] = rmse(predicted-y)
    

#%% EO3 benchmark
EO3prog = sq.fetch_EO3_10oclock_forecast(ts[0], ts[-1])

EO3err = EO3prog - y
sns.jointplot(pd.Series(EO3prog), y)
print "EO3 performance: ",
print rmse(EO3err), mae(EO3err), mape(EO3err,y)