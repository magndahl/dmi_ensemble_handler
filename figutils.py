# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 14:41:35 2016

@author: Magnus Dahl
"""
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import ensemble_tools as ens
import sql_tools as sq

def most_recent_ens_timeseries(start_stop=(dt.datetime(2015,12,16,0), dt.datetime(2016,1,7,0)), pointcode=71699):
    """ star_stop can be a tupple with 2 date tim objects. The first
        is the first time step in the time series, the second is the last.
        
        """
    plt.close('all')    
    weathervars = ['Tout', 'vWind', 'hum', 'sunRad']
    ylabels = ['[$\degree $C]', '[m/s]', '[%]', '[W/m$^2$]']  
    
    suffix = ''.join(['_geo', str(pointcode), '_', ens.timestamp_str(start_stop[0]), \
                        '_to_', ens.timestamp_str(start_stop[1]), '.npy'])
    timesteps = ens.gen_hourly_timesteps(start_stop[0], start_stop[1])
        
    for v, ylab in zip(weathervars, ylabels):
        plt.figure(figsize=(15,20))
        plt.grid(True)
        plt.subplot(2,1,1)        
        ens_data = np.load('time_series/' + v + suffix)
        BBSYD_measured = sq.fetch_BrabrandSydWeather(v, start_stop[0], start_stop[1])
        if v =='Tout':
            ens_data = ens.Kelvin_to_Celcius(ens_data)
        elif v=='hum':
            ens_data = ens.frac_to_percent(ens_data) # convert to percentage                
            
        plt.plot_date(timesteps, ens_data, '-')
        
        plt.plot_date(timesteps, BBSYD_measured, 'k-', lw=2)
        plt.ylabel(ylab)
        plt.grid(True)
        plt.title(v)
        
        plt.subplot(2,1,2)
        plt.plot_date(timesteps, ens.ensemble_std(ens_data), '-', label='Ensemble std')        
        plt.plot_date(timesteps, ens.ensemble_abs_spread(ens_data), '-', label='Max ensemble spread')
        plt.ylabel(ylab)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        figfilename = v + '_most_recent_ens_timeseries.pdf'
        plt.savefig('figures/' + figfilename)