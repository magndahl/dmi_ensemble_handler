# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:07:49 2016

@author: Magnus Dahl
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime as dt
import gurobipy as gb
import ensemble_tools as ens
import sql_tools as sq

PI_T_sup_dict = {'rundhoej':'4.141.120.29.HA.101', 'holme':'4.142.120.29.HA.101', 'hoerning':'4.146.120.29.HA.101'}
PI_Q_dict = {'rundhoej':'H.141.181.02.HA.101', 'holme':'H.142.181.06.HA.101', 'hoerning':'K.146A.181.02.HA.101'}
PI_Q_dict2 = {'holme':'H.142.181.04.HA.101'}

Q_pump_max_dict = {'rundhoej':120, 'holme':430, 'hoerning':410}
Q_max_dict = {k:0.88*Q_pump_max_dict[k] for k in Q_pump_max_dict.keys()}

specific_heat_water = 1.17e-6 # MWh/kgKelvin
density_water = 980 # kg/m3 at 65 deg C
T_ret = 36.5

def get_Tmin_func(df, T_min_q = 0.01, N_Q_q=11):
    
    Q_quantiles = [df['Q'].quantile(i) for i in np.linspace(0,1,N_Q_q)]
    Q_q_midpoints = Q_quantiles[:-1] + 0.5*np.diff(Q_quantiles)
    
    T_min_vals = [df[(df['Q']>Q_quantiles[i]) & (df['Q']<Q_quantiles[i+1])]['T_sup'].quantile(T_min_q) for i in range(N_Q_q-1)]       
    Tmin_func = interp1d(Q_q_midpoints, T_min_vals, kind='linear', fill_value="extrapolate")
        
    return Tmin_func, Q_quantiles


def Tsup_vs_Tout(df, ts1, ts2, station):
    Tout1 = sq.fetch_BrabrandSydWeather('Tout', ts1[0], ts1[-1])
    Tout2 = sq.fetch_BrabrandSydWeather('Tout', ts2[0], ts2[-1])
    Tout = np.concatenate([Tout1, Tout2])
    
    Tout_low_pass = [Tout[range(i-23,i+1)].mean() for i in range(len(Tout))]
    
    plt.figure()
    plt.scatter(Tout_low_pass, df['T_sup'])
    
    slope = -0.74
    Tmin_dict = {'holme':65, 'rundhoej':64, 'hoerning':62}
    Tmin = Tmin_dict[station]
    
    Toutforline = np.linspace(-10, 25, 100)
    
    def Tsuplim(Tout, b):
        return max(Tmin, slope*Tout+b)
        
    def no_points_below(b):
        res = []
        for To, Ts in zip(Tout_low_pass, df['T_sup']):
            if Ts < Tsuplim(To, b):
                res.append(1)
        return len(res)
    
    frac_below = 0.005    
    correct_b = fmin(lambda b:np.abs(no_points_below(b)-frac_below*len(df)), 70)
    print len(df), frac_below*len(df)
    print correct_b
    Tsuplimforline = [Tsuplim(T, correct_b) for T in Toutforline]
    plt.plot(Toutforline, Tsuplimforline)
    
    return
    

def op_model(P, T_min_func, Q_max, T_ret):
    def cons_from_Q(Q):
        return specific_heat_water*density_water*Q*(T_min_func(Q)-T_ret)
        
    Q = fmin(lambda Q:np.abs(cons_from_Q(Q)-P), Q_max/2, disp=False)
       
    if Q<=Q_max:
        T = T_min_func(Q)
        return T, Q
    else:
        Q = Q_max
        T = P/(specific_heat_water*density_water*Q) + T_ret
        return T, Q
        
def detect_outliers(X, station):
    if station=='hoerning':
            outlierfraction = 0.0015
            classifier = svm.OneClassSVM(nu=0.95*outlierfraction + 0.05,
                                         kernel='rbf', gamma=0.1)
            Xscaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
            X_scaled = Xscaler.transform(X)
            classifier.fit(X_scaled)
            svcpred = classifier.decision_function(X_scaled).ravel()
            threshold = stats.scoreatpercentile(svcpred, 100*outlierfraction)
            inlierpred = svcpred>threshold        
            
    else:
        outlierfraction = 0.0015
        classifier = EllipticEnvelope(contamination=outlierfraction)
        classifier.fit(X)
        gausspred = classifier.decision_function(X).ravel()
        threshold = stats.scoreatpercentile(gausspred, 100*outlierfraction)
        inlierpred = gausspred>threshold
            
    return inlierpred

#%%
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
    # old tsstart dt.datetime(2014,12,17,1)
    ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,4,1,1), dt.datetime(2016,1,15,0))
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,4,1,0))
    all_ts = ts1 + ts2
    

    
    PI_T_sup = PI_T_sup_dict[station]
    
    
    if station == 'holme':
        PI_Q1 = PI_Q_dict[station]
        PI_Q2 = PI_Q_dict2[station]
        df = pd.DataFrame()
        df['T_sup']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts1[0], \
                    ts1[-1]),sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts2[0], ts2[-1])])
        df['Q1']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q1, ts1[0], ts1[-1]),\
                    sq.fetch_hourly_vals_from_PIno(PI_Q1, ts2[0], ts2[-1])])
        df['Q2']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q2, ts1[0], ts1[-1]),\
                    sq.fetch_hourly_vals_from_PIno(PI_Q2, ts2[0], ts2[-1])])
        df['Q'] = df['Q1']+df['Q2']
        df['ts'] = all_ts
        df['cons'] = specific_heat_water*density_water*df['Q']*(df['T_sup']-T_ret)
     
    else:
        PI_Q = PI_Q_dict[station]
        df = pd.DataFrame()
        df['T_sup']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts1[0], \
                    ts1[-1]),sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts2[0], ts2[-1])])
        df['Q']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q, ts1[0], ts1[-1]),\
                    sq.fetch_hourly_vals_from_PIno(PI_Q, ts2[0], ts2[-1])])
        
        df['ts'] = all_ts
        df['cons'] = specific_heat_water*density_water*df['Q']*(df['T_sup']-T_ret)
    
    
    #%% outlierdetection
    X = df[['T_sup','Q']]
    outlier_detection = False
    if outlier_detection: 
        detect_outliers(X, station)
    else:
        inlierpred = np.ones(len(df), dtype=bool)
    
    
    #%% The below section only runs if we view Tmin as a function of Q (the old way)
    TminofQ = False
    if TminofQ:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        cond_df = df#[df['Q'] > 288]
        ax1.plot_date(np.array(cond_df['ts']), np.array(cond_df['Q']), 'b')
        
        ax2.plot_date(np.array(cond_df['ts']), np.array(cond_df['T_sup']), 'r-')
        
        plt.figure()
        plt.plot_date(df['ts'], df['cons'], 'g-')
        plt.title(station)
        
        plt.figure()
        plt.scatter(df['T_sup'], df['Q'], c=df['cons'], alpha=0.25)
        plt.colorbar()
        plt.title(station)
        
        
        outliers = df[np.logical_not(inlierpred)]
    
        plt.plot(np.array(outliers['T_sup']), np.array(outliers['Q']), 'ko')
    
    
        
        
        
        #%%
        #plot_Tmin_Q_quantiles(df, inlierpred)
        Q = np.linspace(df[inlierpred]['Q'].min(), df[inlierpred]['Q'].max(), 500)
        qs = [0.001, 0.005, 0.01, 0.02275, 0.05, 0.1]
        for q in qs:
            T_min_func, Q_quantiles = get_Tmin_func(df[inlierpred],T_min_q=q, N_Q_q=21)
            plt.plot(T_min_func(Q), Q, label=str(q), lw=2)
        plt.legend()
        for Q_qua in Q_quantiles:
            plt.axhline(y=Q_qua)
        
    
        #%% P vs Q (T=Tmin(Q))    
    
        T_min_func, Q_quantiles = get_Tmin_func(df, T_min_q=0.02275, N_Q_q=21)
        
        plt.figure()
        plt.plot(Q, T_min_func(Q), 'r', label='Tmin')
        P = specific_heat_water*density_water*Q*(T_min_func(Q)-T_ret)   
        plt.plot(Q, P, 'b', label='Cons')
        plt.xlabel('Q')
        plt.legend()
        
        
        plt.figure()
        simP = df['cons']
        res = [op_model(cons, T_min_func, Q_max=Q_max_dict[station], T_ret=T_ret) for cons in simP]
        simT, simQ = zip(*res)
        plt.scatter(df['T_sup'], df['Q'], c='k', alpha=0.1)
        plt.scatter(simT,simQ,c=simP)
        plt.colorbar()

    
    
if __name__ == "__main__":
    main(sys.argv[1:])