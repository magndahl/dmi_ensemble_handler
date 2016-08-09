# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 10:53:26 2016

@author: azfv1n8
"""

import sys
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ensemble_tools as ens
import sql_tools as sq
from model_selection import linear_map, mlin_regression, gen_all_combinations, summary_to_file, mae, mape, rmse


def main(argv):
    plt.close('all')
    
    try:
        station = argv[0]
        if not station in PI_T_sup_dict.keys():
            print "Wrong station, use rundhoej, holme or hoerning"
            return
    except:
        print "No station provided. Defaults to holme."
        station = 'holme'
        
    print station
    
    plt.close('all')
    #%%
    fit_ts = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    vali_ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    test_ts = ens.gen_hourly_timesteps(dt.datetime(2016,2,5,1), dt.datetime(2016,4,1,0))
    
    all_ts = fit_ts + vali_ts + test_ts
    
    weathervars=['Tout', 'vWind', 'sunRad', 'hum']
    
    fit_data = pd.DataFrame()
    vali_data = pd.DataFrame()            
    test_data = pd.DataFrame()
    
    cons_key = sq.consumption_place_key_dict[station]
    fit_data['cons24h_before'] = sq.fetch_consumption(cons_key, fit_ts[0]+dt.timedelta(days=-1), fit_ts[-1]+dt.timedelta(days=-1))
    vali_data['cons24h_before'] = sq.fetch_consumption(cons_key, vali_ts[0]+dt.timedelta(days=-1), vali_ts[-1]+dt.timedelta(days=-1))
    test_data['cons24h_before'] = sq.fetch_consumption(cons_key, test_ts[0]+dt.timedelta(days=-1), test_ts[-1]+dt.timedelta(days=-1))
    
    fit_data['cons'] = sq.fetch_consumption(cons_key, fit_ts[0], fit_ts[-1])
    vali_data['cons'] = sq.fetch_consumption(cons_key, vali_ts[0], vali_ts[-1])
    test_data['cons'] = sq.fetch_consumption(cons_key, test_ts[0], test_ts[-1])
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
    
    fit_y = fit_data['cons']
    columns = ['cons24h_before', 'Tout24hdiff', 'vWind24hdiff', 'sunRad24hdiff']
    X = fit_data[columns]
    res = mlin_regression(fit_y,X, add_const=False)
    
    fiterr = res.fittedvalues - fit_y
    print "Errors fit period: ", rmse(fiterr), mae(fiterr), mape(fiterr, fit_y)
    
    vali_pred = linear_map(vali_data, res.params, columns)
    valierr = vali_pred - vali_data['cons']
    print "Errors validation period: ", rmse(valierr), mae(valierr), mape(valierr, vali_data['cons'])
    
    test_pred = linear_map(test_data, res.params, columns)
    testerr = test_pred - test_data['cons']
    print "Errors test period: ", rmse(testerr), mae(testerr), mape(testerr, test_data['cons'])
    
    plt.figure()
    plt.plot_date(all_ts, all_data['cons'], 'k-')
    plt.plot_date(all_ts, np.concatenate([res.fittedvalues, vali_pred, test_pred]), 'r-')


if __name__ == "__main__":
    main(sys.argv[1:])