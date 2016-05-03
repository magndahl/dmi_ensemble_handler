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
X = all_data[['prod48hbefore', 'Tout48hdiff', 'prod168hbefore', 'Tout168hdiff', 'vWind168hdiff']]
est = mlin_regression(y, X, add_const=False)

fv = est.fittedvalues
plt.close('all')
sns.jointplot(fv, y)
plt.figure()
plt.plot_date(ts, y, 'b-')
plt.plot_date(ts, fv, 'r-')

err = y-fv
print rmse(err), mae(err), mape(err,y)
err48h_ago = np.roll(err, 48)
all_data['err48h_ago'] = err48h_ago

X = all_data[['prod48hbefore', 'Tout48hdiff', 'prod168hbefore', 'Tout168hdiff', 'vWind168hdiff', 'err48h_ago']]
est2 = mlin_regression(y, X, add_const=False)
fv2 = est2.fittedvalues
err2 = y - est2.fittedvalues

print rmse(err2), mae(err2), mape(err2,y)

sns.jointplot(fv2, y)
plt.figure()
plt.plot_date(ts, y, 'b-')
plt.plot_date(ts, fv2, 'r-')

#%%
all_Z_data = pd.DataFrame()
for c in all_data.columns:
    all_Z_data[c] = (all_data[c]-all_data[c].mean())/all_data[c].std()
X_data = all_Z_data.ix[:, all_Z_data.columns != 'prod']
yz = all_Z_data['prod']
lr = linear_model.LinearRegression()


predicted = cross_val_predict(lr, X_data, yz, cv=25)
plt.figure()
plt.plot(yz)
plt.plot(predicted, 'r')
sns.jointplot(pd.Series(predicted), yz)
score = cross_val_score(lr, X_data, yz, cv=25)

cv = cross_validation.KFold(len(X_data), n_folds=10, shuffle=False, random_state=None)
cv_estimates = []
for train_cv, test_cv in cv:
    cv_estimates.append(mlin_regression(yz[train_cv], X_data.iloc[train_cv], add_const=True))
    
    
with open('cv_res.txt', 'w') as f:
    for e in cv_estimates:
        summary_to_file(e, f)
        

#%% EO3 benchmark
EO3prog = sq.fetch_EO3_10oclock_forecast(ts[0], ts[-1])

EO3err = EO3prog - y
sns.jointplot(pd.Series(EO3prog), y)
print "EO3 performance: ",
print rmse(EO3err), mae(EO3err), mape(EO3err,y)