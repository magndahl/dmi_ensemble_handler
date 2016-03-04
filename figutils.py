# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 14:41:35 2016

@author: Magnus Dahl
"""
import datetime as dt
import operator

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.dates import DateFormatter
from cycler import cycler

import numpy as np
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import ensemble_tools as ens
import sql_tools as sq
from model_selection import linear_map, mlin_regression, mae, mape, rmse

# unicode charecters
uni_tothethird = u'\u00B3'
uni_degree = u'\u00B0'
uni_squared = u'\u00B2'


weathervars = ['Tout', 'vWind', 'hum', 'sunRad']
# column and double column with for Elsevier journals
colwidth = 3.346 # inches
dcolwidth = 6.629 # inches

#colors
blue = '#134b7c'
yellow = '#f8ca00'
orange = '#e97f02'
brown = '#876310'
green = '#4a8e05'
lightgreen = '#b9f73e'#'#c9f76f'
red = '#ae1215'
purple = '#4f0a3d'
darkred= '#4f1215'
pink = '#bd157d'
lightpink = '#d89bc2'
aqua = '#47fff9'
darkblue = '#09233b'
lightblue = '#8dc1e0'
grayblue = '#4a7fa2'
darkgrey = '#333333'

color_cycle = [blue, red, orange, purple, green, pink, lightblue, darkred, yellow, aqua]

def most_recent_ens_timeseries(start_stop=(dt.datetime(2015,12,16,0), dt.datetime(2016,1,19,0)), pointcode=71699, shift_steno_one=False):
    """ star_stop can be a tupple with 2 date tim objects. The first
        is the first time step in the time series, the second is the last.
        
        """
    plt.close('all')    
    ylabels = ['[$\degree $C]', '[m/s]', '[%]', '[W/m$^2$]']  
    
    suffix = ''.join(['_geo', str(pointcode), '_', ens.timestamp_str(start_stop[0]), \
                        '_to_', ens.timestamp_str(start_stop[1]), '.npy'])
    timesteps = ens.gen_hourly_timesteps(start_stop[0], start_stop[1])
    
    Steno_data = np.load('Q:/Weatherdata/Steno_weatherstation/Steno_hourly_2015120111_to_2016011800.npz')
    Steno_Tvhs = Steno_data['Tout_vWind_hum_sunRad']
    Steno_timesteps = Steno_data['timesteps']
        
    for v, ylab in zip(weathervars, ylabels):
        plt.figure(figsize=(15,20))
        plt.grid(True)
        plt.subplot(2,1,1)        
        ens_data = np.load('time_series/' + v + suffix)
        BBSYD_measured = sq.fetch_BrabrandSydWeather(v, start_stop[0], start_stop[1])
        Steno_measured = Steno_Tvhs[:,weathervars.index(v)]
        if shift_steno_one:
            Steno_measured = np.roll(Steno_measured, -1)
        
        if v =='Tout':
            ens_data = ens.Kelvin_to_Celcius(ens_data)
        elif v=='hum':
            ens_data = ens.frac_to_percent(ens_data) # convert to percentage                
        
        
        plt.plot_date(timesteps, ens_data, '-')
        
        plt.plot_date(timesteps, BBSYD_measured, 'k-', lw=2, label='Measured: Brabrand Syd')
        plt.plot_date(Steno_timesteps, Steno_measured, 'r-', lw=2, label='Measured: Steno Museum')
        plt.ylabel(ylab)
        plt.grid(True)
        plt.xlim(start_stop)
        plt.title(v)
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot_date(timesteps, ens.ensemble_std(ens_data), '-', label='Ensemble std')        
        plt.plot_date(timesteps, ens.ensemble_abs_spread(ens_data), '-', label='Max ensemble spread')
        plt.ylabel(ylab)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        figfilename = v + '_most_recent_ens_timeseries.pdf'
        plt.savefig('figures/' + figfilename)
        
def create_5_fold_scatter(avg24=False):
    plt.close('all')
    start_stop=(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    load_suffix = '_geo71699_2015121701_to_2016011500.npy'
    figfilename = 'prod_weather_pairplot.pdf'
    if avg24:
        load_suffix = 'avg24' + load_suffix
        figfilename = 'avg24_' + figfilename
    load_path = 'time_series/ens_means/'
    
    data_dict = {v:np.load(load_path + v + load_suffix) for v in weathervars}
    data_dict['prod'] = sq.fetch_production(start_stop[0], start_stop[1])
    data_dict['(Tout-17)*vWind'] = (data_dict['Tout']-17)*data_dict['vWind']
    
    dataframe = pd.DataFrame(data_dict)
    
    sns.pairplot(dataframe)    
    
    plt.savefig('figures/' + figfilename)

    
def corr_coeff_plot():
    plt.close('all')
    start_stop=(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    load_suffix = '_geo71699_2015121701_to_2016011500.npy'
    load_path = 'time_series/ens_means/'
    
    allvars = weathervars + [v + 'avg24' for v in weathervars]
    data_dict = {v:np.load(load_path + v + load_suffix) for v in allvars}
    data_dict['prod'] = sq.fetch_production(start_stop[0], start_stop[1])
    data_dict['(Tout-17)*vWind'] = (data_dict['Tout']-17)*data_dict['vWind']
    data_dict['(Toutavg-17)*vWindavg24'] = (data_dict['Toutavg24']-17)*data_dict['vWindavg24']
    
    dataframe = pd.DataFrame(data_dict)
    sns.heatmap(dataframe.corr())
    
    return dataframe
    

def check_ens_mean_data():
    plt.close('all')
    start_stop=(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    timesteps = ens.gen_hourly_timesteps(start_stop[0], start_stop[1])
     
    for v in weathervars:
        hourly_data = np.load('time_series/ens_means/' + v +'_geo71699_2015121701_to_2016011500.npy')
        daily_avg_data = np.load('time_series/ens_means/' + v +'avg24_geo71699_2015121701_to_2016011500.npy')
        plt.figure()
        plt.title(v)
        plt.plot_date(timesteps, hourly_data, '-', label='Hourly')
        plt.plot_date(timesteps, daily_avg_data, '-', label='Average over last 24h')
        plt.legend()
               

def check_for_timeshift():
    """ This function chec if there is a time shift between data from
        the Brabrand Syd weather station and the Steno Museum one. 
        It appears that Steno data is one hour fast..
        
        """
    
    plt.close('all')
    start_stop=(dt.datetime(2015,12,16,0), dt.datetime(2016,1,16,0))

    timesteps = ens.gen_hourly_timesteps(start_stop[0], start_stop[1])
    
    Steno_data = np.load('Q:/Weatherdata/Steno_weatherstation/Steno_hourly_2015120111_to_2016011800.npz')
    Steno_Tvhs = Steno_data['Tout_vWind_hum_sunRad']
    Steno_timesteps = Steno_data['timesteps']
    start_index = np.where(Steno_timesteps==start_stop[0])[0]
    end_index = np.where(Steno_timesteps==start_stop[1])[0] + 1
    Steno_Tvhs_short = Steno_Tvhs[start_index:end_index, :]
    Steno_timesteps_new = Steno_timesteps[start_index:end_index]
    assert(all(Steno_timesteps_new==timesteps))
    
    for v in weathervars:
        plt.figure()
        for offset in range(-2,3,1):
            plt.subplot(5,1,offset+3)
            
            BBSYD_measured = sq.fetch_BrabrandSydWeather(v, start_stop[0], start_stop[1])
            Steno_measured = Steno_Tvhs_short[:, weathervars.index(v)]
            Steno_with_offset = np.roll(Steno_measured, offset)
            MAPE = np.mean(np.abs((Steno_with_offset-BBSYD_measured)))
            plt.title('offset %i, MAE = %2.4f '%(offset,MAPE))
            plt.plot_date(timesteps, BBSYD_measured, 'k')
            plt.plot_date(timesteps, Steno_with_offset, 'r')
        
        plt.tight_layout()
        plt.suptitle(v)
            

         
def first_ens_prod_fig():
    """ This plot is based on a production model taking into account:
        Tout, vWind and the production 24 hours before
        
        """
        
    plt.close('all')
    cols = ['Tout', 'vWind', 'prod24h_before']
        
    ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,1,28,0))
    
    #load the data
    fit_data = ens.repack_ens_mean_as_df()
    fit_data['prod24h_before'] = sq.fetch_production(dt.datetime(2015,12,16,1), dt.datetime(2016,1,14,0))

    vali_data = ens.repack_ens_mean_as_df(dt.datetime(2016,1,20,1), dt.datetime(2016,1,28,0))
    vali_data['prod24h_before'] = sq.fetch_production(dt.datetime(2016,1,19,1), dt.datetime(2016,1,27,0))   
    
 
    # do the fit
    X = fit_data[cols]
    y = fit_data['prod']
    res = mlin_regression(y, X, add_const=True)    
    
    fig, [ax1, ax2] = plt.subplots(2,1, figsize=(40,20))
    
    # load ensemble data
    ens_data1 = ens.load_ens_timeseries_as_df(ts_start=ts1[0], ts_end=ts1[-1])
    ens_data1['prod24h_before'] = fit_data['prod24h_before']    
    ens_data2 = ens.load_ens_timeseries_as_df(ts_start=ts2[0], ts_end=ts2[-1])
    ens_data2['prod24h_before'] = vali_data['prod24h_before']
    
    all_ens_data = pd.concat([ens_data1, ens_data2])
    all_ts = ts1 + ts2    
    
    
    # calculate production for each ensemble member
    ens_prods = np.zeros((len(all_ts), 25))
    for i in range(25):
        ens_cols = ['Tout' + str(i), 'vWind' + str(i), 'prod24h_before']
        ens_params = pd.Series({'Tout' + str(i):res.params['Tout'],
                                'vWind' + str(i):res.params['vWind'],
                                'const':res.params['const'],
                                'prod24h_before':res.params['prod24h_before']})
        ens_prods[:,i] = linear_map(all_ens_data, ens_params, ens_cols)    
    
    
       
    # calculate combined confint
    prstd, iv_l, iv_u = wls_prediction_std(res)
    mean_conf_int_spread = np.mean(res.fittedvalues - iv_l)
    model_std = np.concatenate([prstd, (1./1.9599)*mean_conf_int_spread*np.ones(len(ts2))])
    ens_std = ens_prods.std(axis=1)
    combined_std = np.sqrt(model_std**2 + ens_std**2)
    all_prod_model = np.concatenate([res.fittedvalues, linear_map(vali_data, res.params, cols)])
    combined_ub95 = all_prod_model + 1.9599*combined_std
    combined_lb95 = all_prod_model - 1.9599*combined_std 
    
    # plot confint
    ax1.fill_between(all_ts, combined_lb95, combined_ub95, label='Combined 95% conf. int.')
    ax1.fill_between(all_ts, all_prod_model - 1.9599*ens_std, all_prod_model + 1.9599*ens_std, facecolor='grey', label='Ensemble 95% conf. int.')
    
    # plot ensempble models    
    ax1.plot_date(all_ts, ens_prods, '-', lw=0.5)    
    
    ax1.plot_date(ts1, y, 'k-', lw=2, label='Actual production')
    ax1.plot_date(ts1, res.fittedvalues,'r-', lw=2, label='Model on ensemble mean')
         
    ax1.plot_date(ts2, vali_data['prod'], 'k-', lw=2, label='')
    ax1.plot_date(ts2, linear_map(vali_data, res.params, cols), 'r-', lw=2)
    ax1.set_ylabel('[MW]')
    ax1.legend(loc=2)
    
    vali_resid = linear_map(vali_data, res.params, cols) - vali_data['prod']
    ax2.plot_date(ts1, res.resid, '-', label='Residual, fitted data')
    ax2.plot_date(ts2, vali_resid, '-', label='Residual, validation data')
    ax2.set_ylabel('[MW]')
    ax2.legend(loc=2)
    print "MAE = " + str(mae(vali_resid))
    print "MAPE = " + str(mape(vali_resid, vali_data['prod']))
    print "RMSE = " + str(rmse(vali_resid))
    print "ME = " + str(np.mean(vali_resid))
    
    print "MAE (fit) = " + str(mae(res.resid))
    print "MAPE (fit) = " + str(mape(res.resid, fit_data['prod']))
    print "RMSE (fit)= " + str(rmse(res.resid))
    print "ME (fit)= " + str(np.mean(res.resid))

    plt.savefig('figures/ens_prod_models.pdf', dpi=600) 
    plt.figure()
    plt.plot_date(all_ts, ens_std)
    plt.ylabel('Std. of ensemble production models [MW]')
    plt.savefig('figures/std_ens_prod_models.pdf', dpi=600) 
    
    
    sns.jointplot(x=ens_std, y=np.concatenate([res.resid, vali_resid]))
   
        
    return res, all_ens_data, all_ts, fit_data['prod'], vali_data['prod']
    
    
def second_ens_prod_fig():
    """ This plot is based on a production model taking into account:
        the production 24 hours before as well as the change in
        temparature, windspeed and solar radiotion from 24 hours ago to now.
        
        """
        
    plt.close('all')
    cols = ['prod24h_before', 'Tout24hdiff', 'vWind24hdiff', 'sunRad24hdiff']
        
    ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    
    #load the data
    fit_data = ens.repack_ens_mean_as_df()
    fit_data['prod24h_before'] = sq.fetch_production(ts1[0]+dt.timedelta(days=-1), ts1[-1]+dt.timedelta(days=-1))
    
    fit_data['Tout24hdiff'] = fit_data['Tout'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout']).mean(axis=1)
    fit_data['vWind24hdiff'] = fit_data['vWind'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['vWind']).mean(axis=1)
    fit_data['sunRad24hdiff'] = fit_data['sunRad'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['sunRad']).mean(axis=1)
                                    
    vali_data = ens.repack_ens_mean_as_df(ts2[0], ts2[-1])
    vali_data['prod24h_before'] = sq.fetch_production(ts2[0]+dt.timedelta(days=-1), ts2[-1]+dt.timedelta(days=-1))
    vali_data['Tout24hdiff'] = vali_data['Tout'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout']).mean(axis=1)
    vali_data['vWind24hdiff'] = vali_data['vWind'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['vWind']).mean(axis=1)
    vali_data['sunRad24hdiff'] = vali_data['sunRad'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['sunRad']).mean(axis=1)
    
    # correct error in production:
    new_val = (vali_data['prod'][116] +vali_data['prod'][116])/2
    vali_data['prod'][116] = new_val
    vali_data['prod'][117] = new_val
    vali_data['prod24h_before'][116+24] = new_val
    vali_data['prod24h_before'][117+24] = new_val
    
    
 
    # do the fit
    X = fit_data[cols]
    y = fit_data['prod']
    res = mlin_regression(y, X, add_const=False)    
    
    fig, [ax1, ax2] = plt.subplots(2,1, figsize=(40,20))
 
    # load ensemble data
    ens_data1 = ens.load_ens_timeseries_as_df(ts_start=ts1[0], ts_end=ts1[-1],\
                                             weathervars=['Tout', 'vWind', 'sunRad'])
    ens_data1['prod24h_before'] = fit_data['prod24h_before']
    ens_data1_24h_before =  ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout', 'vWind', 'sunRad']) 
    ens_data2 = ens.load_ens_timeseries_as_df(ts_start=ts2[0], ts_end=ts2[-1],\
                                             weathervars=['Tout', 'vWind', 'sunRad'])
    ens_data2['prod24h_before'] = vali_data['prod24h_before']
    ens_data2_24h_before = ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout', 'vWind', 'sunRad']) 
    for i in range(25):
        for v in ['Tout', 'vWind', 'sunRad']:
            key_raw = v + str(i)
            key_diff = v + '24hdiff' + str(i)
            ens_data1[key_diff] = ens_data1[key_raw] - ens_data1_24h_before[key_raw]
            ens_data2[key_diff] = ens_data2[key_raw] - ens_data2_24h_before[key_raw]

    
    all_ens_data = pd.concat([ens_data1, ens_data2])
    all_ts = ts1 + ts2    
#    
#    
    # calculate production for each ensemble member
    ens_prods = np.zeros((len(all_ts), 25))
    for i in range(25):
        ens_cols = ['Tout24hdiff' + str(i), 'vWind24hdiff' + str(i),\
                    'sunRad24hdiff' + str(i), 'prod24h_before']
        ens_params = pd.Series({'Tout24hdiff' + str(i):res.params['Tout24hdiff'],
                                'vWind24hdiff' + str(i):res.params['vWind24hdiff'],
                                'sunRad24hdiff' + str(i):res.params['sunRad24hdiff'],
                                'prod24h_before':res.params['prod24h_before']})
        ens_prods[:,i] = linear_map(all_ens_data, ens_params, ens_cols)    
    
    
       
    # calculate combined confint
    ens_std = ens_prods.std(axis=1)
    vali_resid = linear_map(vali_data, res.params, cols) - vali_data['prod']
    vali_resid_corrig = vali_resid - np.sign(vali_resid)*1.9599*ens_std[len(ts1):]
    mean_conf_int_spread = (vali_resid_corrig.quantile(0.95) - vali_resid_corrig.quantile(0.05))/2
    
    
    combined_conf_int = mean_conf_int_spread + 1.9599*ens_std
    all_prod_model = np.concatenate([res.fittedvalues, linear_map(vali_data, res.params, cols)])
    combined_ub95 = all_prod_model + combined_conf_int
    combined_lb95 = all_prod_model - combined_conf_int 
    
    # plot confint
    ax1.fill_between(all_ts, combined_lb95, combined_ub95, label='Combined 95% conf. int.')
    ax1.fill_between(all_ts, all_prod_model - 1.9599*ens_std, all_prod_model + 1.9599*ens_std, facecolor='grey', label='Ensemble 95% conf. int.')
    
    # plot ensempble models    
    ax1.plot_date(all_ts, ens_prods, '-', lw=0.5)    
    
    ax1.plot_date(ts1, y, 'k-', lw=2, label='Actual production')
    ax1.plot_date(ts1, res.fittedvalues,'r-', lw=2, label='Model on ensemble mean')
         
    ax1.plot_date(ts2, vali_data['prod'], 'k-', lw=2, label='')
    ax1.plot_date(ts2, linear_map(vali_data, res.params, cols), 'r-', lw=2)
    ax1.set_ylabel('[MW]')
    ax1.legend(loc=2)
    ax1.set_ylim([0,1100])
    
    
    ax2.plot_date(ts1, res.resid, '-', label='Residual, fitted data')
    ax2.plot_date(ts2, vali_resid, '-', label='Residual, validation data')
    ax2.set_ylabel('[MW]')
    ax2.legend(loc=2)
    ax2.set_ylim([-550, 550])
    print "MAE = " + str(mae(vali_resid))
    print "MAPE = " + str(mape(vali_resid, vali_data['prod']))
    print "RMSE = " + str(rmse(vali_resid))
    print "ME = " + str(np.mean(vali_resid))
    
    print "MAE (fit) = " + str(mae(res.resid))
    print "MAPE (fit) = " + str(mape(res.resid, fit_data['prod']))
    print "RMSE (fit)= " + str(rmse(res.resid))
    print "ME (fit)= " + str(np.mean(res.resid))

    plt.savefig('figures/ens_prod_models_v2.pdf', dpi=600) 
    plt.figure()
    plt.plot_date(all_ts, ens_std)
    plt.ylabel('Std. of ensemble production models [MW]')
    plt.savefig('figures/std_ens_prod_models.pdf', dpi=600) 
    # 
    
    vali_ens_std = ens_std[len(ts1):]
    sns.jointplot(x=pd.Series(vali_ens_std), y=np.abs(vali_resid))
    sns.jointplot(x=vali_data['prod'], y=pd.Series(linear_map(vali_data, res.params, cols)))
   
    EO3_fc1 = sq.fetch_EO3_midnight_forecast(ts1[0], ts1[-1])
    EO3_fc2 = sq.fetch_EO3_midnight_forecast(ts2[0], ts2[-1])
    plt.figure()
    plt.plot_date(ts1, fit_data['prod'], 'k-', label='Actual production')
    plt.plot_date(ts2, vali_data['prod'], 'k-')
    plt.plot_date(ts1, EO3_fc1, 'r-', label='EO3 forecast')
    plt.plot_date(ts2, EO3_fc2, 'r-')
    EO3_err = EO3_fc2-vali_data['prod']
    EO3_err_fit = EO3_fc1-fit_data['prod']
    print "MAE (EO3) = " + str(mae(EO3_err))
    print "MAPE (EO3) = " + str(mape(EO3_err, vali_data['prod']))
    print "RMSE (EO3)= " + str(rmse(EO3_err))
    print "ME (EO3)= " + str(np.mean(EO3_err))
    
    print "MAE (EO3_fit) = " + str(mae(EO3_err_fit))
    print "MAPE (EO3_fit) = " + str(mape(EO3_err_fit, fit_data['prod']))
    print "RMSE (EO3_fit)= " + str(rmse(EO3_err_fit))
    print "ME (EO3_fit)= " + str(np.mean(EO3_err_fit))
     
    sns.jointplot(x=pd.Series(vali_ens_std), y=np.abs(EO3_err))
    
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.plot_date(all_ts, combined_conf_int/combined_conf_int.max(), '-')
    plt.ylabel('Model + ensemble uncertainty \n [normalized]')
    plt.ylim(0,1)    
    plt.subplot(2,1,2)
    plt.plot_date(all_ts, (1-0.2*combined_conf_int/combined_conf_int.max()), '-', label='Dynamic setpoint')
    plt.plot_date(all_ts, 0.8*np.ones(len(all_ts)), '--', label='Static setpoint')
    plt.ylabel('Setpoint for pump massflow \n temperature [fraction of max pump cap]')
    plt.legend()
    plt.ylim(.7,1)
    plt.savefig('figures/setpoint.pdf')

    
    return vali_data, fit_data, res, ens_std, vali_resid


## these functions are for the first article and the comment show where they belong (fig2, fig3, fig 4 etc)
def weather_forecast_ensemble(): # figure 2
    plt.close('all')
    ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    ens_data = ens.load_ens_timeseries_as_df(ts_start=ts[0], ts_end=ts[-1],\
                                             weathervars=['Tout', 'vWind', 'sunRad'])
    fig, axes = plt.subplots(3,1, sharex=True, figsize=(colwidth, 1.65*colwidth))
    plt.xticks(size=5)
    
    ylabels = [u'Outside temperature [%sC]'%uni_degree, 'Wind speed [m/s]', u'Solar irradiance [W/m%s]'%uni_squared]
    
    for  ax, v, cshift, ylab in zip(axes, ['Tout', 'vWind', 'sunRad'], (15,23,6), ylabels):
        color_list = plt.cm.Dark2(np.roll(np.linspace(0, 1, 25), cshift))        
        ax.set_prop_cycle(cycler('color',color_list))
        v_ens_data = ens_data[[v + str(i) for i in range(25)]]
        ax.plot_date(ts, v_ens_data, '-', lw=0.5)
        ax.set_ylabel(ylab, size=8)
        ax.tick_params(axis='y', which='major', labelsize=8)
        plt.box(True)
    plt.tight_layout()
    axes[-1].xaxis.set_major_formatter(DateFormatter('%b %d') )
    axes[-1].set_xlim(dt.datetime(2016,1,20,0), dt.datetime(2016,2,5,0))
    fig.savefig('figures/first_articlefigs/weather_forecast_ensemble.pdf')
    return ens_data, axes


def production_model(): # figure 3
    
    plt.close('all')
    cols = ['prod24h_before', 'Tout24hdiff', 'vWind24hdiff', 'sunRad24hdiff']
        
    ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    
    #load the data
    fit_data = ens.repack_ens_mean_as_df()
    fit_data['prod24h_before'] = sq.fetch_production(ts1[0]+dt.timedelta(days=-1), ts1[-1]+dt.timedelta(days=-1))
    
    fit_data['Tout24hdiff'] = fit_data['Tout'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout']).mean(axis=1)
    fit_data['vWind24hdiff'] = fit_data['vWind'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['vWind']).mean(axis=1)
    fit_data['sunRad24hdiff'] = fit_data['sunRad'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['sunRad']).mean(axis=1)
                                    
    vali_data = ens.repack_ens_mean_as_df(ts2[0], ts2[-1])
    vali_data['prod24h_before'] = sq.fetch_production(ts2[0]+dt.timedelta(days=-1), ts2[-1]+dt.timedelta(days=-1))
    vali_data['Tout24hdiff'] = vali_data['Tout'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout']).mean(axis=1)
    vali_data['vWind24hdiff'] = vali_data['vWind'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['vWind']).mean(axis=1)
    vali_data['sunRad24hdiff'] = vali_data['sunRad'] \
                                - ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['sunRad']).mean(axis=1)
    
    # correct error in production:
    new_val = (vali_data['prod'][116] +vali_data['prod'][116])/2
    vali_data['prod'][116] = new_val
    vali_data['prod'][117] = new_val
    vali_data['prod24h_before'][116+24] = new_val
    vali_data['prod24h_before'][117+24] = new_val
    
    
 
    # do the fit
    X = fit_data[cols]
    y = fit_data['prod']
    res = mlin_regression(y, X, add_const=False)    

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(dcolwidth, 0.55*dcolwidth), gridspec_kw={'height_ratios':[4,1]})

    # load ensemble data
    ens_data1 = ens.load_ens_timeseries_as_df(ts_start=ts1[0], ts_end=ts1[-1],\
                                             weathervars=['Tout', 'vWind', 'sunRad'])
    ens_data1['prod24h_before'] = fit_data['prod24h_before']
    ens_data1_24h_before =  ens.load_ens_timeseries_as_df(\
                                    ts_start=ts1[0]+dt.timedelta(days=-1),\
                                    ts_end=ts1[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout', 'vWind', 'sunRad']) 
    ens_data2 = ens.load_ens_timeseries_as_df(ts_start=ts2[0], ts_end=ts2[-1],\
                                             weathervars=['Tout', 'vWind', 'sunRad'])
    ens_data2['prod24h_before'] = vali_data['prod24h_before']
    ens_data2_24h_before = ens.load_ens_timeseries_as_df(\
                                    ts_start=ts2[0]+dt.timedelta(days=-1),\
                                    ts_end=ts2[-1]+dt.timedelta(days=-1), \
                                    weathervars=['Tout', 'vWind', 'sunRad']) 
    for i in range(25):
        for v in ['Tout', 'vWind', 'sunRad']:
            key_raw = v + str(i)
            key_diff = v + '24hdiff' + str(i)
            ens_data1[key_diff] = ens_data1[key_raw] - ens_data1_24h_before[key_raw]
            ens_data2[key_diff] = ens_data2[key_raw] - ens_data2_24h_before[key_raw]

    
    all_ens_data = pd.concat([ens_data1, ens_data2])
    all_ts = ts1 + ts2    
#    
#    
    # calculate production for each ensemble member
    ens_prods = np.zeros((len(all_ts), 25))
    for i in range(25):
        ens_cols = ['Tout24hdiff' + str(i), 'vWind24hdiff' + str(i),\
                    'sunRad24hdiff' + str(i), 'prod24h_before']
        ens_params = pd.Series({'Tout24hdiff' + str(i):res.params['Tout24hdiff'],
                                'vWind24hdiff' + str(i):res.params['vWind24hdiff'],
                                'sunRad24hdiff' + str(i):res.params['sunRad24hdiff'],
                                'prod24h_before':res.params['prod24h_before']})
        ens_prods[:,i] = linear_map(all_ens_data, ens_params, ens_cols)    
    
    
       
    # calculate combined confint
    ens_std = ens_prods.std(axis=1)
    vali_resid = linear_map(vali_data, res.params, cols) - vali_data['prod']
    vali_resid_corrig = vali_resid - np.sign(vali_resid)*1.9599*ens_std[len(ts1):]
    #mean_conf_int_spread = (vali_resid_corrig.quantile(0.95) - vali_resid_corrig.quantile(0.05))/2 # this conf_int is not used anymore


    fit_resid = res.resid
    fit_resid_corrig = fit_resid - np.sign(fit_resid)*1.9599*ens_std[0:len(ts1)]
    conf_int_spread_lower = - fit_resid_corrig.quantile(0.025)
    conf_int_spread_higher = fit_resid_corrig.quantile(0.975) 
    
    combined_conf_ints = conf_int_spread_lower + conf_int_spread_higher + 2*1.9599*ens_std
    all_prod_model = np.concatenate([res.fittedvalues, linear_map(vali_data, res.params, cols)])
    combined_ub95 = all_prod_model + conf_int_spread_higher + 1.9599*ens_std
    combined_lb95 = all_prod_model - (conf_int_spread_lower + 1.9599*ens_std)
    
    # plot confint
    ax1.fill_between(all_ts[len(ts1):], combined_lb95[len(ts1):], combined_ub95[len(ts1):], label='95% prediction intervals')
    ax1.fill_between(all_ts[len(ts1):], all_prod_model[len(ts1):] - 1.9599*ens_std[len(ts1):], all_prod_model[len(ts1):] + 1.9599*ens_std[len(ts1):], facecolor='grey', label='Weather ensemble 95% conf. int.')
    
    # plot ensempble models    
    ax1.plot_date(all_ts[len(ts1):], ens_prods[len(ts1):], '-', lw=0.5)    

    ax1.plot_date(ts2, vali_data['prod'], 'k-', lw=2, label='Historical production')
    ax1.plot_date(ts2, linear_map(vali_data, res.params, cols), '-', c=red, lw=2, label='Production model')
    ax1.set_ylabel('Production [MW]', size=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.xaxis.set_major_formatter(DateFormatter('%b %d') )    
    ax1.legend(loc=1, prop={'size':8})
    ax1.set_ylim([300,1100])
    
    N = conf_int_spread_higher + 1.9599*ens_std[len(ts1):].max()
    ax2.fill_between(ts2, -(1.9599*ens_std[len(ts1):]+conf_int_spread_lower)/N, -1.9599*ens_std[len(ts1):]/N, alpha=0.5)
    ax2.fill_between(ts2, -1.9599*ens_std[len(ts1):]/N, np.zeros(len(ts2)), facecolor='grey',alpha=0.5)
    ax2.fill_between(ts2, 1.9599*ens_std[len(ts1):]/N, facecolor='grey')
    ax2.fill_between(ts2, 1.9599*ens_std[len(ts1):]/N, (conf_int_spread_higher+1.9599*ens_std[len(ts1):])/N) 
    ax2.set_ylabel('Prediction intervals \n[normalized]', size=8)
    ax2.tick_params(axis='y', which='major', labelsize=8)
    ax2.set_xlim(dt.datetime(2016,1,20,0), dt.datetime(2016,2,5,0))
    fig.tight_layout()
    print "Min_normalized pos conf bound. ", np.min(1.9599*ens_std[len(ts1):]/N+conf_int_spread_higher/N)
    
    print "MAE = " + str(mae(vali_resid))
    print "MAPE = " + str(mape(vali_resid, vali_data['prod']))
    print "RMSE = " + str(rmse(vali_resid))
    print "ME = " + str(np.mean(vali_resid))
    
    print "MAE (fit) = " + str(mae(res.resid))
    print "MAPE (fit) = " + str(mape(res.resid, fit_data['prod']))
    print "RMSE (fit)= " + str(rmse(res.resid))
    print "ME (fit)= " + str(np.mean(res.resid))
    
    print "Width of const blue bands (MW)", conf_int_spread_lower, conf_int_spread_higher

    plt.savefig('figures/first_articlefigs/production_model.pdf', dpi=400) 

   
    EO3_fc1 = sq.fetch_EO3_midnight_forecast(ts1[0], ts1[-1])
    EO3_fc2 = sq.fetch_EO3_midnight_forecast(ts2[0], ts2[-1])
    EO3_err = EO3_fc2-vali_data['prod']
    EO3_err_fit = EO3_fc1-fit_data['prod']
    print "MAE (EO3) = " + str(mae(EO3_err))
    print "MAPE (EO3) = " + str(mape(EO3_err, vali_data['prod']))
    print "RMSE (EO3)= " + str(rmse(EO3_err))
    print "ME (EO3)= " + str(np.mean(EO3_err))
    
    print "MAE (EO3_fit) = " + str(mae(EO3_err_fit))
    print "MAPE (EO3_fit) = " + str(mape(EO3_err_fit, fit_data['prod']))
    print "RMSE (EO3_fit)= " + str(rmse(EO3_err_fit))
    print "ME (EO3_fit)= " + str(np.mean(EO3_err_fit))
    
    print np.min(combined_conf_ints[len(ts1):]/combined_conf_ints.max())
    np.savez('combined_conf_int', combined_conf_int=(conf_int_spread_higher+1.9599*ens_std), timesteps=all_ts)

    print "Corr coeff: vali ", np.corrcoef(vali_data['prod'],linear_map(vali_data, res.params, cols))[0,1]
    print "Corr coeff: vali EO3 ", np.corrcoef(vali_data['prod'], EO3_fc2)[0,1]
    print "Corr coeff: fit ", np.corrcoef(fit_data['prod'],res.fittedvalues)[0,1]
    print "Corr coeff: fit EO3 ", np.corrcoef(fit_data['prod'], EO3_fc1)[0,1]
    
    print "% of actual production in vali period above upper", float(len(np.where(vali_data['prod']>(conf_int_spread_higher+1.9599*ens_std[len(ts1):]+linear_map(vali_data, res.params, cols)))[0]))/len(ts2)
    print "plus minus: ", 0.5/len(ts2)
    
    print "% of actual production in vali period below lower", float(len(np.where(vali_data['prod']<(linear_map(vali_data, res.params, cols)-(conf_int_spread_lower+1.9599*ens_std[len(ts1):])))[0]))/len(ts2)
    print "plus minus: ", 0.5/len(ts2)
    
    return res, fit_data
    
    
def hoerning_pump_model(): # figure 4
    # simple model
    T1 = 68.5
    a2 = 15.5
    a3 = 2.1
    b2 = 295-a2*T1
    b3 = 340-a3*71.4
    
    def Q_from_cons_lin_piece(cons, a, b):
        B = -(b+a*T_ret)/a
        C = -cons/(specific_heat_water*density_water)
        A = 1/a
    
        Qplus = (-B+np.sqrt(B**2 - 4*A*C))/(2*A)
        
        return Qplus
    
    def get_Tsup_and_Q(cons, Q_ub):
        # try lowes possible T    
        Q = cons/(specific_heat_water*density_water*(T1 - T_ret))
        if Q <= 295:
            return T1, Q
        elif Q > 295:
            Q = Q_from_cons_lin_piece(cons, a2, b2)
            if Q <= Q_ub*(340./360):
                T = (Q - b2)/a2  
                return T, Q
            elif Q >= Q_ub*(340./360):
                b3_adjusted = b3 + (Q_ub*(340./360) - 340)
                Q = Q_from_cons_lin_piece(cons, a3, b3_adjusted)
                if Q <= Q_ub:
                    T = (Q - b3_adjusted)/a3
                    return T, Q
                elif Q > Q_ub:
                    Q = Q_ub
                    T = cons/(specific_heat_water*density_water*Q) + T_ret
                    return T, Q
                
    plt.close('all')

    fig, [ax1, ax2] = plt.subplots(2,1,sharex=True, sharey=True)
    ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    all_ts = ts1 + ts2
    PI_T_sup = '4.146.120.29.HA.101'
    PI_Q = 'K.146A.181.02.HA.101'
    specific_heat_water = 1.17e-6 # MWh/kgKelvin
    density_water = 980 # kg/m3 at 65 deg C
    T_ret = 36.5
    df = pd.DataFrame()
    df['T_sup']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts1[0], \
            ts1[-1]),sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts2[0], ts2[-1])])
    df['Q']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q, ts1[0], ts1[-1]),\
            sq.fetch_hourly_vals_from_PIno(PI_Q, ts2[0], ts2[-1])])
    df['ts'] = all_ts
    df['cons'] = specific_heat_water*density_water*df['Q']*(df['T_sup']-T_ret)
    
    
    model_conf_int = np.load('combined_conf_int.npz')['combined_conf_int']
    assert(list(np.load('combined_conf_int.npz')['timesteps'])==all_ts), "confidence intervals don't have matching time steps"
    const_Q_ub = 360
    Q_const_cap = []
    T_sup_const_cap = []
    Q_dyn_cap = []
    T_sup_dyn_cap = []
    dyn_Q_ub = []
    for c, model_uncertainty in zip(df['cons'], model_conf_int):
        T_const, Q_const = get_Tsup_and_Q(c, const_Q_ub)
        Q_const_cap.append(Q_const)
        T_sup_const_cap.append(T_const)

        Q_ub = 410 - (410-const_Q_ub)*(model_uncertainty/np.max(model_conf_int))
        dyn_Q_ub.append(Q_ub)
        T_dyn, Q_dyn = get_Tsup_and_Q(c, Q_ub)
        Q_dyn_cap.append(Q_dyn)
        T_sup_dyn_cap.append(T_dyn)
        
    
    dT=0.1
    ax1.fill_between([65+dT,95-dT], [410, 410], [360, 360], facecolor=red, alpha=0.25)
    ax1.fill_between([65+dT,95-dT], [360, 360],[340, 340], facecolor=yellow, alpha=0.25)
    ax1.fill_between([T1, 71.4, 80.9, 100], [295, 340, 360, 360], color='k', edgecolor='k', alpha=0.2, linewidth=1)
    ax2.fill_between([65+dT,95-dT], [410, 410], [360, 360], facecolor=red, alpha=0.25)
    ax2.fill_between([65+dT,95-dT], [360, 360],[340, 340], facecolor=yellow, alpha=0.25)
    ax2.fill_between([T1, 71.4, 80.9, 100], [295, 340, 360, 360], color='k', edgecolor='k', alpha=0.2, linewidth=1)
    ax1.plot([65+dT,95-dT], [410, 410], '--', c=red, lw=2)
    ax1.text(79,415, 'Maximum pump capacity', size=8)
    im = ax1.scatter(T_sup_const_cap, Q_const_cap, c=df['cons'], cmap=plt.cm.BuPu)
    
    ax2.scatter(T_sup_dyn_cap, Q_dyn_cap, c=df['cons'], cmap=plt.cm.BuPu)
    ax2.plot([65+dT,95-dT], [410, 410], '--', c=red, lw=2)
    ax2.text(79,415, 'Maximum pump capacity', size=8)
    
    cax, kw = mpl.colorbar.make_axes([ax1, ax2])
    fig.colorbar(im, cax=cax)
    cax.set_ylabel('Delivered heat [MW]',size=8)

    ax2.set_xlabel(u'Supply temperature [%sC]'%uni_degree, size=8)
    ax1.set_ylabel(u'Water flow rate [m%s/h]'%uni_tothethird, size=8)
    ax2.set_ylabel(u'Water flow rate [m%s/h]'%uni_tothethird, size=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    cax.tick_params(axis='y', which='major', labelsize=8)
    ax1.set_title('Scenario 1', size=10)
    ax2.set_title('Scenario 2', size=10)

    ax1.set_xlim((65,95))
    ax1.set_ylim((150,450))
    
    
    fig.set_size_inches(1.15*colwidth,1.6*colwidth)

    fig.savefig('figures/first_articlefigs/hoerning_pump_model.pdf')
    
    # This is a theoretical calculation in case the model uncertainty was 50% of what it is
    statistical_conf_int = 50.90285 # this number is printed when production_model() is run (Width of const blue band (MW) ...)    
    Q_dyn_cap_half_model_unc = []
    T_sup_dyn_cap_half_model_unc = []
    dyn_Q_ub_half_model_unc = []
    reduced_model_conf_int =  model_conf_int-0.5*statistical_conf_int
    for c, model_uncertainty in zip(df['cons'], reduced_model_conf_int):
        Q_ub = 410 - (410-const_Q_ub)*(model_uncertainty/np.max(model_conf_int))
        dyn_Q_ub_half_model_unc.append(Q_ub)
        T_dyn, Q_dyn = get_Tsup_and_Q(c, Q_ub)
        Q_dyn_cap_half_model_unc.append(Q_dyn)
        T_sup_dyn_cap_half_model_unc.append(T_dyn)
            
    
    return T_sup_const_cap, T_sup_dyn_cap, Q_const_cap, Q_dyn_cap, model_conf_int, T_sup_dyn_cap_half_model_unc
    
    
def Q_T_heatloss_timeseries(): # figure 5
    T_sup_const_cap, T_sup_dyn_cap, Q_const_cap, Q_dyn_cap, model_conf_int, T_sup_dyn_cap_half_model_unc = hoerning_pump_model()
    plt.close('all')
    
    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, figsize=(dcolwidth, 0.55*dcolwidth), gridspec_kw={'height_ratios':[2,1,1]})
    #fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, figsize=(dcolwidth, 0.55*dcolwidth))
    
    ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    
    
    red_area_lb1 = 410 - (410-360)*(model_conf_int[0:len(ts1)]/np.max(model_conf_int))
    red_area_lb2 = 410 - (410-360)*(model_conf_int[len(ts1):]/np.max(model_conf_int))
    yellow_area_lb1 = (340./360)*red_area_lb1
    yellow_area_lb2 = (340./360)*red_area_lb2
    limlw = 0.75
    ax1.plot_date(ts1, red_area_lb1, '-', c=darkgrey, lw=limlw, label='Scenario 2 security margins')
    ax1.plot_date(ts2, red_area_lb2, '-', c=darkgrey, lw=limlw)
    ax1.plot_date(ts1, yellow_area_lb1, '-', c=darkgrey, lw=limlw)
    ax1.plot_date(ts2, yellow_area_lb2, '-', c=darkgrey, lw=limlw)
    ax1.fill_between(ts1, 360*np.ones(len(ts1)), 410*np.ones(len(ts1)), facecolor=red, alpha=0.25)
    ax1.fill_between(ts2, 360*np.ones(len(ts2)), 410*np.ones(len(ts2)), facecolor=red, alpha=0.25)
    ax1.fill_between(ts1, 340*np.ones(len(ts1)), 360*np.ones(len(ts1)), facecolor=yellow, alpha=0.25)
    ax1.fill_between(ts2, 340*np.ones(len(ts2)), 360*np.ones(len(ts2)), facecolor=yellow, alpha=0.25)      
    ax1.plot_date(ts1, Q_const_cap[0:len(ts1)], '-', c=red, label='Scenario 1')
    ax1.plot_date(ts2, Q_const_cap[len(ts1):], '-', c=red)
    ax1.plot_date(ts1, Q_dyn_cap[0:len(ts1)], '-', c=green, lw=1, label='Scenario 2')
    ax1.plot_date(ts2, Q_dyn_cap[len(ts1):], '-', c=green, lw=1)    
    ax1.plot_date(ts1+ts2, 410*np.ones(len(ts1+ts2)), '--', c=red, lw=1)
    handles, labels = ax1.get_legend_handles_labels()
    hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)

    ax1.legend(handles2, labels2, loc=0, prop={'size':8})


    ax2.plot_date(ts1, T_sup_const_cap[0:len(ts1)], '-', c=red, label='Scenario 1')
    ax2.plot_date(ts2, T_sup_const_cap[len(ts1):], '-', c=red)
    ax2.plot_date(ts1, T_sup_dyn_cap[0:len(ts1)], '-', c=green, lw=1, label='Scenario 2')
    ax2.plot_date(ts2, T_sup_dyn_cap[len(ts1):], '-', c=green, lw=1)
    ax2.legend(loc=6, prop={'size':8})
   
    T_grnd = 6.4
    heat_loss_reduction = 100*(1 - (np.array(T_sup_dyn_cap) - T_grnd)/(np.array(T_sup_const_cap) - T_grnd))
    heat_loss_reduction_half_model_unc = 100*(1 - (np.array(T_sup_dyn_cap_half_model_unc) - T_grnd)/(np.array(T_sup_const_cap) - T_grnd))

    redu_heat_loss1 = heat_loss_reduction[0:len(ts1)]
    redu_heat_loss2 = heat_loss_reduction[len(ts1):]
    ax3.plot_date(ts1, redu_heat_loss1, '-', c=blue, lw=1)
    ax3.plot_date(ts2, redu_heat_loss2, '-', c=blue, lw=1)
    ax3.xaxis.set_major_formatter(DateFormatter('%b %d \n %Y') )
    ax1.tick_params(axis='y', which='major', labelsize=8)
    ax1.set_ylim(150,450)
    ax2.tick_params(axis='y', which='major', labelsize=8)
    ax3.tick_params(axis='y', which='major', labelsize=8)
    ax1.set_ylabel(u'Flow rate  [m%s/h]'%uni_tothethird, size=8)
    ax2.set_ylabel(u'Supply\ntemperature [%sC]'%uni_degree, size=8)
    ax3.set_ylabel('Heat loss\nreduction [%]', size=8)
    
    mjloc = mpl.ticker.MultipleLocator(1)
    ax3.yaxis.set_major_locator(mjloc)
    ax3.set_xlim(dt.datetime(2015,12,17,0), dt.datetime(2016,2,5,0))
    fig.tight_layout()
    
    fig.savefig('figures/first_articlefigs/Q_T_heatloss_timeseries.pdf')
    
    return heat_loss_reduction, heat_loss_reduction_half_model_unc
    