# -*- coding: utf-8 -*-
"""

Created on Thu Jan 21 09:50:42 2016

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

import pandas as pd


all_data = ens.repack_ens_mean_as_df()

hours = [np.mod(h, 24) for h in range(1,697)]

all_data['prod24h_before'] = sq.fetch_production(dt.datetime(2015,12,16,1), dt.datetime(2016,1,14,0))
all_data['(Tout-17)*vWind*hum'] = all_data['(Tout-17)*vWind']*all_data['hum']
all_data['(Toutavg24-17)*vWindavg24*humavg24'] = all_data['(Toutavg-17)*vWindavg24']*all_data['humavg24']
all_data['Tout24hdiff'] = all_data['Tout'] - np.roll(all_data['Tout'], 24)
Tout24h_before = ens.load_ens_timeseries_as_df(ts_start=dt.datetime(2015,12,16,1),\
                         ts_end=dt.datetime(2016,1,14,0), weathervars=['Tout']).mean(axis=1)
vWind24h_before = ens.load_ens_timeseries_as_df(ts_start=dt.datetime(2015,12,16,1),\
                         ts_end=dt.datetime(2016,1,14,0), weathervars=['vWind']).mean(axis=1)
sunRad24h_before = ens.load_ens_timeseries_as_df(ts_start=dt.datetime(2015,12,16,1),\
                         ts_end=dt.datetime(2016,1,14,0), weathervars=['sunRad']).mean(axis=1)
hum24h_before = ens.load_ens_timeseries_as_df(ts_start=dt.datetime(2015,12,16,1),\
                         ts_end=dt.datetime(2016,1,14,0), weathervars=['hum']).mean(axis=1)
                         
all_data['Tout24hdiff'] = all_data['Tout'] - Tout24h_before
all_data['vWind24hdiff'] = all_data['vWind'] - vWind24h_before
all_data['sunRad24hdiff'] = all_data['sunRad'] - sunRad24h_before
all_data['sunRadavg2424hdiff'] = all_data['sunRadavg24'] - np.roll(all_data['sunRadavg24'],24)
all_data['hum24hdiff'] = all_data['hum'] - hum24h_before

for c in all_data.columns:
    all_data['Z' + c] = (all_data[c]-all_data[c].mean())/all_data[c].std()


y = all_data['prod']

def mlin_regression(y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X) # adds a constant term to the regression
    est = sm.OLS(y,X).fit()
    return est
    
    
def summary_to_file(est, openfile):
    openfile.write(est.summary().as_text())
    

def gen_all_combinations(columns=['(Tout-17)*vWind', '(Toutavg-17)*vWindavg24', 'Toutavg24',
       'hum', 'humavg24', 'sunRad', 'sunRadavg24', 'vWind',
       'vWindavg24']):
    
    subsets = []       
    for length in range(1, len(columns)+1):
        for subset in combinations(columns, length):
            subsets.append(list(subset))
                        
    return subsets
    
def include_Tout(subsets, Tout_str='Tout'):
    """ include one combination with only Tout and include Tout in all other
        combs.
        
        """
        
    subsets_with_Tout = [[Tout_str] + s for s in subsets]
    return [[Tout_str]] + subsets_with_Tout
    
def include_prod24h_before(subsets, prod24hbefor_str='prod24h_before'):
    return [[prod24hbefor_str] + s for s in subsets]
    

def make_all_fits(all_combs=include_Tout(gen_all_combinations()), add_const=True, y=y):
    results = []
    for columns in all_combs:
        X = all_data[columns]
        res = mlin_regression(y,X, add_const=add_const)
        results.append(res)
    
    return results
    
    
def norm_xTx(X):
    X = np.array(X)
    norm_x = np.ones_like(X)
    for i in range(X.shape[1]):
        norm_x[:, i] = X[:, i] / np.linalg.norm(X[:, i])
    norm_xTx = np.dot(norm_x.T, norm_x)
    
    return norm_xTx
    
    
def condition_number(X):
    normxTx = norm_xTx(X)
    eigs = np.linalg.eigvals(normxTx)
    condition_number = np.sqrt(eigs.max() / eigs.min())
    
    return condition_number
    
    
def save_all_fits(savefile='all_fit_summary.txt', \
    all_combs=include_Tout(gen_all_combinations()), add_const=True, y=y):
    results = make_all_fits(all_combs, add_const, y=y)
    
    with open(savefile,'w') as myfile:   
        for c, res in zip(all_combs, results):
            cond_number = res.condition_number
            if all(res.pvalues < 0.05) and cond_number<20:
                myfile.write("\n-------------------- \n Condition number = %2.3f \n"%cond_number)
                myfile.write("MAE = " + str(mae(res.resid)) + " \n")
                myfile.write("RMSE = " + str(rmse(res.resid)) + " \n------------------- \n")
                summary_to_file(res,myfile)
                
    return results
    
            

def save_good_fit_candidates(savefile='good_fit_summary.txt'):
    combs = [['Tout', 'Toutavg24', '(Tout-17)*vWind'],\
              ['Tout', 'Toutavg24','vWind'],\
              ['Tout', 'Toutavg24', '(Tout-17)*vWind', 'vWindavg24'],\
              ['Tout', 'Toutavg24', 'vWind', 'vWindavg24'],
              ['Tout', 'Toutavg24', 'vWind', 'vWindavg24', 'sunRadavg24']]
     
    results = make_all_fits(combs)          
    with open(savefile,'w') as myfile:   
       for c, res in zip(combs, results):
           cond_number = res.condition_number
           my_cond_number = condition_number(all_data[c])
           
           myfile.write(\
           "\n-------------------- \n Condition number = %2.3f \n My condition number = %2.3f ------------------- \n"%(cond_number, my_cond_number))
           summary_to_file(res,myfile)
                
    
def plot_best_model():
    plt.close('all')
    columns = ['Tout', 'Toutavg24', 'vWind', 'vWindavg24']#, 'hours', 'hours2','hours3', 'hours4','hours5', 'hours6']#, 'hours7', 'hours8']#,'hours5', 'hours6']
    X = all_data[columns]
    res = mlin_regression(y, X)
    timesteps = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    
    plt.subplot(2,1,1)
    plt.plot_date(timesteps, y, 'b', label='Actual prodution')
    plt.plot_date(timesteps, res.fittedvalues, 'r', label='Weather model')
    prstd, iv_l, iv_u = wls_prediction_std(res)    
    plt.plot_date(timesteps, iv_u, 'r--', label='95% conf. int.')
    plt.plot_date(timesteps, iv_l, 'r--')
    mean_day_resid = [res.resid[i::24].mean() for i in range(24)]
    mean_resid_series = np.tile(mean_day_resid, 29)
    plt.plot_date(timesteps, res.fittedvalues + mean_resid_series, 'g', label='Weather model + avg daily profile')
    plt.ylabel('MW')
    plt.legend(loc=2)
    plt.subplot(2,1,2)
    plt.plot_date(timesteps, res.resid, '-', label='Residual')
    
    plt.plot_date(timesteps, mean_resid_series)
    plt.ylabel('MW')
    plt.legend()
    
    mape = np.mean(np.abs((res.fittedvalues + mean_resid_series-y)/y))
    mape2 = np.mean(np.abs((res.resid)/y))
    mae = np.mean(np.abs((res.fittedvalues + mean_resid_series-y)))
    
    print mape, mape2, mae
    
    
    res.summary()
    return res

def save_best_model():
    columns = ['Tout', 'Toutavg24', 'vWind', 'vWindavg24']
    X = all_data[columns]
    res = mlin_regression(y, X)        
    
    res.params.to_pickle('lin_reg_fit_params.pkl')
    mean_day_resid = [res.resid[i::24].mean() for i in range(24)]
    
    np.save('daily_profile.npy', mean_day_resid)
    
    return res
    

def linear_map(dataframe, params, cols):
    result = np.zeros(len(dataframe))
    for c in cols:
        result += dataframe[c]*params[c]
    
    try:
        result += params['const']
    except:
        print "no constant term in params"
                
    return result
    
    
def validate_ToutToutavg24vWindvWindavg24_model():
    plt.close('all')
    
    ts_start = dt.datetime(2016,1,19,1)
    ts_end = dt.datetime(2016,1,26,0)
    
    daily_profile = np.load('daily_profile.npy')
    params = pd.read_pickle('lin_reg_fit_params.pkl')
    validation_data = ens.repack_ens_mean_as_df(ts_start, ts_end)    
    
    weather_model = linear_map(validation_data, params, ['Tout', 'Toutavg24', 'vWind', 'vWindavg24'])
    timesteps = ens.gen_hourly_timesteps(ts_start, ts_end)
    
    plt.plot_date(timesteps, validation_data['prod'],'b-')
    plt.plot_date(timesteps, weather_model,'r-')
    
    weather_model_wdailyprofile = []
    for ts, wm in zip(timesteps, weather_model):
        print ts.hour
        weather_model_wdailyprofile.append(wm + daily_profile[np.mod(ts.hour-1,24)])
    
    plt.plot_date(timesteps, weather_model_wdailyprofile, 'g-')
    
    return validation_data
    
    
def validate_prod24h_before_and_diffsmodel():
    plt.close('all')
    
    cols = ['prod24h_before', 'Tout24hdiff', 'vWind24hdiff', 'sunRad24hdiff']
    ts_start = dt.datetime(2016,1,20,1)
    ts_end = dt.datetime(2016,1,31,0)
    
    validation_data = ens.repack_ens_mean_as_df(ts_start, ts_end)
    
    # correct error in production:
    new_val = (validation_data['prod'][116] +validation_data['prod'][116])/2
    validation_data['prod'][116] = new_val
    validation_data['prod'][117] = new_val
    validation_data['prod24h_before'] = sq.fetch_production(ts_start+dt.timedelta(days=-1), ts_end+dt.timedelta(days=-1))
    validation_data['prod24h_before'][116+24] = new_val
    validation_data['prod24h_before'][117+24] = new_val
    Tout24h_before = ens.load_ens_timeseries_as_df(ts_start+dt.timedelta(days=-1),\
                         ts_end+dt.timedelta(days=-1), weathervars=['Tout']).mean(axis=1)
    vWind24h_before = ens.load_ens_timeseries_as_df(ts_start+dt.timedelta(days=-1),\
                         ts_end+dt.timedelta(days=-1), weathervars=['vWind']).mean(axis=1)
    sunRad24h_before = ens.load_ens_timeseries_as_df(ts_start+dt.timedelta(days=-1),\
                         ts_end+dt.timedelta(days=-1), weathervars=['sunRad']).mean(axis=1)    
    validation_data['Tout24hdiff'] = validation_data['Tout'] - Tout24h_before
    validation_data['vWind24hdiff'] = validation_data['vWind'] - vWind24h_before
    validation_data['sunRad24hdiff'] = validation_data['sunRad'] - sunRad24h_before
    
    # fit on fit area
    X = all_data[cols]
    res = mlin_regression(all_data['prod'], X, add_const=False)
    
    #apply to validation area
    weather_model = linear_map(validation_data, res.params, cols)
    timesteps = ens.gen_hourly_timesteps(ts_start, ts_end)
    
    plt.plot_date(timesteps, validation_data['prod'],'b-')
    plt.plot_date(timesteps, weather_model,'r-')
    residual = weather_model - validation_data['prod']
    
    return validation_data, res, residual
    

def try_prod24h_before(columns=['Tout', 'vWind', 'vWindavg24', 'prod24h_before'], add_const=False, y=y):
    plt.close('all')
    X = all_data[columns]
    res = mlin_regression(y, X, add_const=add_const)
    timesteps = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    
    plt.subplot(2,1,1)
    plt.plot_date(timesteps, y, 'b', label='Actual prodution')
    plt.plot_date(timesteps, res.fittedvalues, 'r', label='Weather model')
    prstd, iv_l, iv_u = wls_prediction_std(res)    
    plt.plot_date(timesteps, iv_u, 'r--', label='95% conf. int.')
    plt.plot_date(timesteps, iv_l, 'r--')
    plt.ylabel('MW')
    plt.legend(loc=2)
    plt.subplot(2,1,2)
    plt.plot_date(timesteps, res.resid, '-', label='Residual')
    plt.ylabel('MW')
    plt.legend()
    
    print "MAE = " + str(mae(res.resid))
    print "MAPE = " + str(mape(res.resid, y))
    print "RMSE = " + str(rmse(res.resid))
    
    print res.summary()
    
       
    return res
    

def mae(error):
    return np.mean(np.abs(error))
    

def mape(error, prod):
    return mae(error/prod)
    
    
def rmse(error):
    return np.sqrt(np.mean(error**2))
    

def param_from_zscoreparam(zparams, paramname='Tout'):
    if paramname=='const':
        const = all_data['prod'].mean()
        for zpn in zparams.keys():
            pn = zpn[1:] # remove the Z
            const -= all_data[pn].mean()*param_from_zscoreparam(zparams, pn)
        return const
        
    return zparams['Z' + paramname]*all_data['prod'].std()/all_data[paramname].std()


def params_from_zscoreparam(zparams):
    params = pd.Series()
    for zpn in zparams.keys():
        pn = zpn[1:] # remove the Z
        params[pn] = param_from_zscoreparam(zparams, pn)
    params['const'] = param_from_zscoreparam(zparams, 'const')

    return params
    
    
def conf_int_from_zscoreconfint(conf_int, paramname='Tout'):
    if paramname=='const':
        const_lb = all_data['prod'].mean()
        const_ub = all_data['prod'].mean()
        for zpn in conf_int[0].keys():
            pn = zpn[1:] # remove the Z
            const_lb -= all_data[pn].mean()*conf_int_from_zscoreconfint(conf_int, pn)[0]
            const_ub -= all_data[pn].mean()*conf_int_from_zscoreconfint(conf_int, pn)[1]
        return (const_lb, const_ub)
    
    lb = conf_int[0]['Z' + paramname]*all_data['prod'].std()/all_data[paramname].std()
    ub = conf_int[1]['Z' + paramname]*all_data['prod'].std()/all_data[paramname].std()
    
    return (lb,ub)