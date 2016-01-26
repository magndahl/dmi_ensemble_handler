# -*- coding: utf-8 -*-
"""

Created on Thu Jan 21 09:50:42 2016

@author: Magnus Dahl
"""

from itertools import combinations
import statsmodels.api as sm
import datetime as dt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
import matplotlib.pyplot as plt
import ensemble_tools as ens
import pandas as pd


all_data = ens.repack_ens_mean_as_df()

hours = [np.mod(h, 24) for h in range(1,697)]
all_data['hours'] = hours
all_data['hours2'] = [h**2 for h in hours]
all_data['hours3'] = [h**3 for h in hours]
all_data['hours4'] = [h**4 for h in hours]
all_data['hours5'] = [h**5 for h in hours]
all_data['hours6'] = [h**6 for h in hours]
all_data['hours7'] = [h**7 for h in hours]
all_data['hours8'] = [h**8 for h in hours]

y = all_data['prod']

def mlin_regression(y, X):
    X = sm.add_constant(X) # adds a constant term to the regression
    est = sm.OLS(y,X).fit()
    return est
    
    
def summary_to_file(est, openfile):
    openfile.write(est.summary().as_text())
    

def gen_all_combinations(columns=['(Tout-17)*vWind', '(Toutavg-17)*vWindavg24', 'Toutavg24',
       'hum', u'humavg24', 'sunRad', 'sunRadavg24', 'vWind',
       'vWindavg24']):
    
    subsets = []       
    for length in range(1, len(columns)+1):
        for subset in combinations(columns, length):
            subsets.append(list(subset))
                        
    return subsets
    
def include_Tout(subsets):
    """ include one combination with only Tout and include Tout in all other
        combs.
        
        """
        
    subsets_with_Tout = [['Tout'] + s for s in subsets]
    return [['Tout']] + subsets_with_Tout
    

def make_all_fits(all_combs=include_Tout(gen_all_combinations())):
    results = []
    for columns in all_combs:
        X = all_data[columns]
        res = mlin_regression(y,X)
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
    
    
def save_all_fits(savefile='all_fit_summary.txt'):
    all_combs=include_Tout(gen_all_combinations())
    results = make_all_fits(all_combs)
    
    with open(savefile,'w') as myfile:   
        for c, res in zip(all_combs, results):
            cond_number = res.condition_number
            if cond_number < 1e3 and all(res.pvalues <0.05):
                myfile.write("\n-------------------- \n Condition number = %2.3f \n------------------- \n"%cond_number)
                summary_to_file(res,myfile)
            

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
    
