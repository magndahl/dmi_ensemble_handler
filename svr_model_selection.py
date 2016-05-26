# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:17:49 2016

@author: Magnus Dahl
"""

import pandas as pd
import datetime as dt
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_predict
from model_selection import gen_all_combinations, rmse, mae, mape
import sql_tools as sq
import ensemble_tools as ens


#%% SVR experinment


ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,26,1), dt.datetime(2016,4,1,0))
X = pd.read_pickle('48h60h168h_lagged_X.pkl') # run model_selection_ext_horizon to generate these files
y = pd.read_pickle('prod_to_gowith.pkl') 
# add more predictor data:


for v in ['Tout', 'vWind', 'hum', 'sunRad']:
    X[v] = ens.load_ens_mean_avail_at10_series(v, ts[0], ts[-1], pointcode=71699)

#X['weekdays'] = [t.weekday() for t in ts]
def h_hoursbefore(timestamp, h):
    return timestamp + dt.timedelta(hours=-h)
most_recent_avail_prod = sq.fetch_production(h_hoursbefore(ts[0], 24),\
                                                          h_hoursbefore(ts[-1], 24))

for i, t, p48 in zip(range(len(most_recent_avail_prod)), ts, X['prod48hbefore']):
    if t.hour >= 10 or t.hour == 0:
        most_recent_avail_prod[i] = p48

        
X['prod24or48hbefore'] = most_recent_avail_prod
##

X_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
X_scaled = X_scaler.transform(X)
y_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(y)
y_scaled = y_scaler.transform(y)
#%%


svr_rmses = {}
svr_poly_rmses = {}
Cs = np.linspace(14,18,10)
gams = np.linspace(0.003, 0.004, 10)
# SVR with RBF kernel
for C in Cs:
    for gamma in gams:
        svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma)
        y_rbf = cross_val_predict(svr_rbf, X_scaled, y_scaled, cv=10)

        svr_rmses[(C,gamma)] = rmse(y_scaler.inverse_transform(y_rbf)-y)

#%%
svr_rbf = SVR(kernel='rbf', C=16, gamma=0.004)
y_rbf = cross_val_predict(svr_rbf, X_scaled, y_scaled, cv=10)
svr_rmse_lone = rmse(y_scaler.inverse_transform(y_rbf)-y)
svr_mae_lone = mae(y_scaler.inverse_transform(y_rbf)-y)
svr_mape_lone = mape(y_scaler.inverse_transform(y_rbf)-y,y)

lr = linear_model.LinearRegression()

y_lin = cross_val_predict(lr, X_scaled, y_scaled, cv=10)

#%% load ensemble data
def gen_ens_dfs(ts_start, ts_end, varnames, timeshifts, pointcode=71699):
    """ timeshifts must be integer number of hours. Posetive values only,
        dataframe contains columns with the variables minus their value
        'timeshift' hours before. """
    
    
    df = pd.DataFrame()
    
    df_s = [pd.DataFrame() for i in range(25)]
    for timeshift in timeshifts:
        
        prod_before = sq.fetch_production(h_hoursbefore(ts_start, timeshift),\
                                                          h_hoursbefore(ts_end, timeshift))
        for df in df_s:
            df['prod%ihbefore'%timeshift] = prod_before
            
        for v in varnames:
            ens_data = ens.load_ens_avail_at10_series(ts_start, ts_end, v, pointcode=71699)
            ens_data_before = ens.load_ens_avail_at10_series(h_hoursbefore(ts_start, timeshift),\
                                                        h_hoursbefore(ts_end, timeshift), v, pointcode=71699)
            diff = ens_data - ens_data_before
            for i in range(ens_data.shape[1]):
                df_s[i]['%s%ihdiff%i'%(v,timeshift, i)] = diff[:,i]
    for v in varnames:
        ens_data = ens.load_ens_avail_at10_series(ts_start, ts_end, v, pointcode=71699)
        for i in range(ens_data.shape[1]):
            df_s[i]['%s%i'%(v, i)] = ens_data[:,i]         

    for df in df_s:    
        df['prod24or48hbefore'] = most_recent_avail_prod    
    
    return df_s

ens_X_data = gen_ens_dfs(ts[0], ts[-1], ['Tout', 'vWind', 'hum', 'sunRad'],[48, 60, 168])

svr_rbf.fit(X_scaled,y_scaled)
ens_y_data = [svr_rbf.predict(X_scaler.transform(x)) for x in ens_X_data]
ens_ydata_scaled = [y_scaler.inverse_transform(y) for y in ens_y_data]

#%% Test on blind period!
ts_test = ens.gen_hourly_timesteps(dt.datetime(2016,4,1,1), dt.datetime(2016,5,1,0))
X_test = pd.read_pickle('48h60h168h_lagged_X_test.pkl')
X_test = X_test.ix[:, X_test.columns !='prod'] # run model_selection_ext_horizon to generate these files
y_test = pd.read_pickle('prod_to_gowith_test.pkl') 
# add more predictor data:


for v in ['Tout', 'vWind', 'hum', 'sunRad']:
    X_test[v] = ens.load_ens_mean_avail_at10_series(v, ts_test[0], ts_test[-1], pointcode=71699)


most_recent_avail_prod_test = sq.fetch_production(h_hoursbefore(ts_test[0], 24),\
                                                          h_hoursbefore(ts_test[-1], 24))

for i, t, p48 in zip(range(len(most_recent_avail_prod_test)), ts_test, X_test['prod48hbefore']):
    if t.hour >= 10 or t.hour == 0:
        most_recent_avail_prod[i] = p48

        
X_test['prod24or48hbefore'] = most_recent_avail_prod_test
#%%
test_pred = svr_rbf.predict(X_scaler.transform(X_test))
test_err = y_scaler.inverse_transform(test_pred) - y_test

print "SVR model performance"
print rmse(test_err), mae(test_err), mape(test_err,y_test)
