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
from model_selection import gen_all_combinations, mlin_regression, rmse, mae, mape
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

reload_data = True
if reload_data:        
    timelags = [48, 60, 168]
    all_data = gen_fit_df(dt.datetime(2016,1,26,1), dt.datetime(2016,4,1,0), ['Tout', 'vWind', 'hum', 'sunRad'], timelags)
y = all_data['prod']

#%%
ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,26,1), dt.datetime(2016,4,1,0))
X = all_data.ix[:, all_data.columns !='prod']

X.to_pickle('48h60h168h_lagged_X.pkl')
y.to_pickle('prod_to_gowith.pkl')


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
#%%

var_combs = [v for v in var_combs if 'Tout48hdiff' in v and 'prod48hbefore' in v] # this makes for fewer combinations
try_all_combs=True
if try_all_combs:
    maes = np.zeros(len(var_combs))
    rmses = np.zeros(len(var_combs))
    for v, i in zip(var_combs, range(len(var_combs))):
        predicted = cross_val_predict(lr, all_data[v], y, cv=10)
        maes[i] = mae(predicted-y)
        rmses[i] = rmse(predicted-y)
    




#%% EO3 benchmark
EO3prog = sq.fetch_EO3_10oclock_forecast(ts[0], ts[-1])

EO3err = EO3prog - y
sns.jointplot(pd.Series(EO3prog), y)
print "EO3 performance: ",
print rmse(EO3err), mae(EO3err), mape(EO3err,y)


#%% Model selected, now test:
lr.fit(all_data[var_combs[rmses.argmin()]], y)
test_data = gen_fit_df(dt.datetime(2016,4,1,1), dt.datetime(2016,5,1,0), ['Tout', 'vWind', 'hum', 'sunRad'], timelags)

test_predicted = lr.predict(test_data[var_combs[rmses.argmin()]])
test_prod = test_data['prod']

test_data.to_pickle('48h60h168h_lagged_X_test.pkl')
test_prod.to_pickle('prod_to_gowith_test.pkl')

test_err = test_predicted-test_prod
print "My model performance test:"
print rmse(test_err), mae(test_err), mape(test_err,test_prod)

#%% EO3 benchmark on test period 
EO3prog_test = sq.fetch_EO3_10oclock_forecast(dt.datetime(2016,4,1,1), dt.datetime(2016,5,1,0))

EO3err_test = EO3prog_test - test_prod
sns.jointplot(pd.Series(EO3prog_test), test_prod)
print "EO3 performance test: ",
print rmse(EO3err_test), mae(EO3err_test), mape(EO3err_test,test_prod)

#%% ens_data on full period 
def gen_ens_df(ts_start, ts_end, varnames, timeshifts, pointcode=71699):
    """ timeshifts must be integer number of hours. Posetive values only,
        dataframe contains columns with the variables minus their value
        'timeshift' hours before. """
    
    
    df = pd.DataFrame()
    df['prod'] = sq.fetch_production(ts_start, ts_end)
    
    for timeshift in timeshifts:
        
        df['prod%ihbefore'%timeshift] = sq.fetch_production(h_hoursbefore(ts_start, timeshift),\
                                                          h_hoursbefore(ts_end, timeshift))
        for v in varnames:
            ens_data = ens.load_ens_avail_at10_series(ts_start, ts_end, v, pointcode=71699)
            ens_data_before = ens.load_ens_avail_at10_series(h_hoursbefore(ts_start, timeshift),\
                                                        h_hoursbefore(ts_end, timeshift), v, pointcode=71699)
            diff = ens_data - ens_data_before
            for i in range(ens_data.shape[1]):
                df['%s%ihdiff%i'%(v,timeshift, i)] = diff[:,i]
    
    
    return df

long_ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,26,1), dt.datetime(2016,5,1,0))
ens_prediction = []
ens_data = gen_ens_df(dt.datetime(2016,1,26,1), dt.datetime(2016,5,1,0), ['Tout', 'vWind'], timelags)

prod_vars = ['prod48hbefore', 'prod60hbefore', 'prod168hbefore']
weather_vars = ['Tout48hdiff', 'Tout168hdiff', 'vWind168hdiff']


lr.fit(all_data[var_combs[rmses.argmin()]], y)
for i in range(25):
    all_vars = prod_vars + [v+str(i) for v in weather_vars]
    all_vars[1], all_vars[3] = all_vars[3], all_vars[1]
    all_vars[2], all_vars[3] = all_vars[3], all_vars[2]## This restores the righ position of the variables
      
    ens_prediction.append(lr.predict(ens_data[all_vars]))

#%%        
plt.figure()
for i in range(25):
    plt.plot_date(long_ts, ens_prediction[i], '-')
    
plt.plot_date(long_ts, ens_data['prod'], 'k-', lw=2, label='Actual production')
plt.ylabel('Production [MW]')

ens_pred_std = np.array(ens_prediction).std(axis=0)
plt.legend()

#%% Estimating model uncertainty
test_resid_corrig = test_err - np.sign(test_err)*1.9599*ens_pred_std[-len(test_prod):]
mean_conf_int_spread = (test_resid_corrig.quantile(0.975) - test_resid_corrig.quantile(0.025))/2

test_resid_corrig_68 = test_err - np.sign(test_err)*ens_pred_std[-len(test_prod):]
mean_conf_int_spread_68 = (test_resid_corrig.quantile(0.6826) - test_resid_corrig.quantile(1-0.6826))/2

long_data_means = gen_fit_df(dt.datetime(2016,1,26,1), dt.datetime(2016,5,1,0), ['Tout', 'vWind'], timelags)


#%% Plots for Grethe and Jeanette
plt.figure()
plt.plot_date(ts, all_data['prod'], 'k-', lw=2, label='Actual production')
plt.plot_date(ts, lr.predict(all_data[var_combs[rmses.argmin()]]), 'b-', label='My model')
plt.plot_date(ts, EO3prog, 'r-', label='EO3 forecast')
plt.ylabel('Production [MW]')
plt.legend()

long_predict = lr.predict(long_data_means[var_combs[rmses.argmin()]])

plt.figure()

ub = long_predict + mean_conf_int_spread + 1.9599*ens_pred_std
lb = long_predict - mean_conf_int_spread - 1.9599*ens_pred_std
ub68 = long_predict + mean_conf_int_spread_68 + ens_pred_std
lb68 = long_predict - mean_conf_int_spread_68 - ens_pred_std
plt.fill_between(long_ts, lb, ub, facecolor='pink', label='95% prediction interval')
plt.fill_between(long_ts, lb68, ub68, facecolor='red', label='68% prediction interval')
plt.plot_date(long_ts, long_data_means['prod'], 'k', lw=2, label='Actual production')
plt.plot_date(long_ts, long_predict, 'b', lw=3, label='My model')
plt.ylabel('Production [MW]')
plt.legend()