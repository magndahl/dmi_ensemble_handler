# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:11:50 2016

@author: Magnus Dahl
"""

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import ensemble_tools as ens
import sql_tools as sq

plt.close('all')

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]
    
 
def autocorr2(x, lag=1):
    rho = np.corrcoef(x, np.roll(x,lag))[0,1]
    
    return  rho
    

def my_diff(x, lag=24):
    return x-np.roll(x,lag)

ts = ens.gen_hourly_timesteps(dt.datetime(2013, 1, 1, 1), dt.datetime(2016,1,1,0))
prod = sq.fetch_production(ts[0], ts[-1])

norm_prod = (prod-prod.mean())/prod.std()

plt.plot_date(ts, prod, '-')


auto_c = autocorr(norm_prod)

rho_i = [autocorr2(prod, i) for i in range(2*168)]

prod_24h_diff = my_diff(prod)

rho2 =  [autocorr2(prod_24h_diff, i) for i in range(2*168)]
prod_48h_diff = my_diff(prod, 48)

rho3 = [autocorr2(prod_48h_diff, i) for i in range(2*168)]
plt.figure()
plt.plot(rho3, 'r')


#%% Smooth prod

s_prod = gaussian_filter1d(prod, 3*168)

plt.figure()
plt.plot_date(ts, prod)
plt.plot_date(ts, s_prod)

plt.plot_date(ts, prod-s_prod, 'r')

#%% Fourier analysis of production:

plt.figure()
amps = np.abs(np.fft.rfft(prod, norm='ortho'))
freq = np.fft.rfftfreq(len(prod))
plt.subplot(2,1,1)
plt.semilogx(freq, amps)
plt.subplot(2,1,2)
plt.semilogx(1/freq, amps)
