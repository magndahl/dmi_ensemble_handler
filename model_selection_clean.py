# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:04:19 2016

@author: azfv1n8
"""
import datetime as dt
import numpy as np
import pandas as pd

import ensemble_tools as ens
import sql_tools as sq
from model_selection import linear_map, mlin_regression, gen_all_combinations, summary_to_file, mae, mape, rmse

#%%
fit_ts = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
vali_ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
test_ts = ens.gen_hourly_timesteps(dt.datetime(2016,2,5,1), dt.datetime(2016,3,1,0))

all_ts = fit_ts + vali_ts + test_ts

weathervars=['Tout', 'vWind', 'sunRad', 'hum']

fit_data = pd.DataFrame()
vali_data = pd.DataFrame()            
test_data = pd.DataFrame()
                
fit_data['prod24h_before'] = sq.fetch_production(fit_ts[0]+dt.timedelta(days=-1), fit_ts[-1]+dt.timedelta(days=-1))
vali_data['prod24h_before'] = sq.fetch_production(vali_ts[0]+dt.timedelta(days=-1), vali_ts[-1]+dt.timedelta(days=-1))
test_data['prod24h_before'] = sq.fetch_production(test_ts[0]+dt.timedelta(days=-1), test_ts[-1]+dt.timedelta(days=-1))

fit_data['prod'] = sq.fetch_production(fit_ts[0], fit_ts[-1])
vali_data['prod'] = sq.fetch_production(vali_ts[0], vali_ts[-1])
test_data['prod'] = sq.fetch_production(test_ts[0], test_ts[-1])
for v in weathervars:
    fit_data['%s24hdiff'%v] = ens.load_ens_timeseries_as_df(\
                                ts_start=fit_ts[0],\
                                ts_end=fit_ts[-1], \
                                weathervars=[v]).mean(axis=1) \
                              - ens.load_ens_timeseries_as_df(\
                                ts_start=fit_ts[0]+dt.timedelta(days=-1),\
                                ts_end=fit_ts[-1]+dt.timedelta(days=-1), \
                                weathervars=[v]).mean(axis=1)
    vali_data['%s24hdiff'%v] = ens.load_ens_timeseries_as_df(\
                                ts_start=vali_ts[0],\
                                ts_end=vali_ts[-1], \
                                weathervars=[v]).mean(axis=1) \
                              - ens.load_ens_timeseries_as_df(\
                                ts_start=vali_ts[0]+dt.timedelta(days=-1),\
                                ts_end=vali_ts[-1]+dt.timedelta(days=-1), \
                                weathervars=[v]).mean(axis=1)
    test_data['%s24hdiff'%v] = ens.load_ens_timeseries_as_df(\
                                ts_start=test_ts[0],\
                                ts_end=test_ts[-1], \
                                weathervars=[v]).mean(axis=1) \
                              - ens.load_ens_timeseries_as_df(\
                                ts_start=test_ts[0]+dt.timedelta(days=-1),\
                                ts_end=test_ts[-1]+dt.timedelta(days=-1), \
                                weathervars=[v]).mean(axis=1)
                                
                                
#%%
all_data = pd.concat([fit_data, vali_data, test_data])
no_blind_data = pd.concat([fit_data, vali_data])

corr = no_blind_data.corr()

#%% Try fitting all combinations
all_combs = gen_all_combinations(all_data.drop(['prod', 'prod24h_before'], axis=1).columns)
for c in all_combs:
    c.insert(0,'prod24h_before')
all_combs.insert(0, ['prod24h_before'])

check_AIC=False
if check_AIC:
    for c in fit_data.columns:
        fit_data[c] = (fit_data[c]-fit_data[c].mean())/fit_data[c].std()

fit_y = fit_data['prod']
results = []
for columns in all_combs:
        X = fit_data[columns]
        res = mlin_regression(fit_y,X, add_const=False)
        results.append(res)

vali_preds = []
for cols in all_combs:
    vali_pred = linear_map(vali_data, res.params, cols)
    vali_preds.append(vali_pred)

rmses = [rmse(vp-vali_data['prod']) for vp in vali_preds]
aics = [r.aic for r in results]

for c,r,a in zip(all_combs, rmses, aics):
    print c,r,a
    
right_columns = ['prod24h_before', 'Tout24hdiff', 'vWind24hdiff', 'sunRad24hdiff']
test_pred = linear_map(test_data, results[all_combs.index(right_columns)].params, right_columns)
print "Test RMSE", rmse(test_pred-test_data['prod'])
print "Test MAE", mae(test_pred-test_data['prod'])
print "Test MAPE", mape(test_pred-test_data['prod'], test_data['prod'])

EO3_fc_test = sq.fetch_EO3_midnight_forecast(test_ts[0], test_ts[-1])
EO3_err = EO3_fc_test-test_data['prod']

print "MAE (EO3) = " + str(mae(EO3_err))
print "MAPE (EO3) = " + str(mape(EO3_err, test_data['prod']))
print "RMSE (EO3)= " + str(rmse(EO3_err))
print "ME (EO3)= " + str(np.mean(EO3_err))