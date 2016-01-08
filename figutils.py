# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 14:41:35 2016

@author: Magnus Dahl
"""
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import ensemble_tools as ens

def most_recent_ens_timeseries(start_stop=None, pointcode=71699):
    """ star_stop can be a tupple with 2 date tim objects. The first
        is the first time step in the time series, the second is the last.
        
        """
    plt.close('all')    
    weathervars = ['Tout', 'vWind', 'hum', 'sunRad']
    ylabels = ['[$\degree $C]', '[m/s]', '[]', '[$W/m^2$]']
    if start_stop == None:
        suffix = '_geo71699_2015121600_to_2016010700.npy'
        timesteps = ens.gen_hourly_timesteps(dt.datetime(2015,12,16,0), dt.datetime(2016,1,7,0))
    else:
        suffix = ''.join(['_geo', str(pointcode), '_', ens.timestamp_str(start_stop[0]), \
                        '_to_', ens.timestamp_str(start_stop[1])])
        timesteps = ens.gen_hourly_timesteps(start_stop[0], start_stop[1])
        
    for v, ylab in zip(weathervars, ylabels):
        plt.figure(figsize=(15,20))
        plt.grid(True)
        plt.subplot(2,1,1)        
        data = np.load('time_series/' + v + suffix)
        if v =='Tout':
            data = ens.Kelvin_to_Celcius(data)
        plt.plot_date(timesteps, data, '-')
        plt.ylabel(ylab)
        plt.grid(True)
        
        plt.subplot(2,1,2)
        plt.plot_date(timesteps, ens.ensemble_std(data), '-', label='Ensemble std')        
        plt.plot_date(timesteps, ens.ensemble_abs_spread(data), '-', label='Max ensemble spread')
        plt.ylabel(ylab)
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(v)
        plt.tight_layout()
        figfilename = v + '_most_recent_ens_timeseries.pdf'
        plt.savefig('figures/' + figfilename)