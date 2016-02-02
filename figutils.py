# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 14:41:35 2016

@author: Magnus Dahl
"""
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ensemble_tools as ens
import sql_tools as sq
from model_selection import linear_map, mlin_regression, mae, mape, rmse
from statsmodels.sandbox.regression.predstd import wls_prediction_std


weathervars = ['Tout', 'vWind', 'hum', 'sunRad']


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
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,1,31,0))
    
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
    vali_resid = linear_map(vali_data, res.params, cols) - vali_data['prod']
    mean_conf_int_spread = (vali_resid.quantile(0.95) - vali_resid.quantile(0.05))/2
    model_std = (1./1.9599)*mean_conf_int_spread*np.ones(len(all_ts))
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
    
    return vali_data, fit_data, res, combined_std
