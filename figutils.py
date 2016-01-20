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
            
            
    