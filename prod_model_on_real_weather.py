# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 15:15:03 2016

@author: Magnus Dahl
"""

import statsmodels as sm
import sql_tools as sq
import numpy as np
import pandas as pd
import datetime as dt
from model_selection import mlin_regression, mae, mape, rmse
from ensemble_tools import gen_hourly_timesteps
import matplotlib.pyplot as plt
plt.close('all')

def check_fit(res):
    correct_signs = {'Tout24hdiff':-1, 'sunRad24hdiff':-1, 'vWind24hdiff':1, 'hum24hdiff':1, 'prod24h_before':1}
    for var in res.pvalues.index:
        if res.pvalues[var] > 0.03:
            print res.pvalues[var], var
            return False, var
        elif correct_signs[var]*res.params[var] < 0:
            return False, var
                
    if np.abs(res.params['prod24h_before']-1) > 0.05:
        print "WARNING: prod24h_before is weighted with: " + str(res.params['prod24h_before'])
    if res.resid.mean()>5:
        print "WARNING: Bias in model: " + res.resid.mean()
    return True, None
    

ts_start = dt.datetime(2015, 10, 17, 1)
ts_end = dt.datetime(2016,1,16,0)
timesteps = gen_hourly_timesteps(ts_start, ts_end)
df = pd.DataFrame()

df['prod'] = sq.fetch_production(ts_start, ts_end)
df['prod24h_before'] = sq.fetch_production(ts_start + dt.timedelta(days=-1), \
                                            ts_end + dt.timedelta(days=-1))
                                            
for v in ['Tout', 'vWind', 'sunRad', 'hum']:
    df[v] = sq.fetch_BrabrandSydWeather(v, ts_start, ts_end)
    df[v + '24h_before'] = sq.fetch_BrabrandSydWeather(v, ts_start + dt.timedelta(days=-1), \
                                            ts_end + dt.timedelta(days=-1))
    df[v + '24hdiff'] = df[v] - df[v + '24h_before']
                                            
cols = ['Tout24hdiff', 'vWind24hdiff', 'prod24h_before', 'sunRad24hdiff', 'hum24hdiff']
good_fit = False
while not good_fit:
    X = df[cols]
    res = mlin_regression(df['prod'], X, add_const=False)
    print res.summary()    
    good_fit, problem_var = check_fit(res)
    try:
        cols.remove(problem_var)
    except:
        print "Final cols were: " + str(cols)


plt.plot_date(timesteps, df['prod'], '-k')
plt.plot_date(timesteps, res.fittedvalues, '-r')

print "MAE (fit) = " + str(mae(res.resid))
print "MAPE (fit) = " + str(mape(res.resid, df['prod']))
print "RMSE (fit)= " + str(rmse(res.resid))
print "ME (fit)= " + str(np.mean(res.resid))


