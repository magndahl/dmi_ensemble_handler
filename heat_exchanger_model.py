# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:07:49 2016

@author: Magnus Dahl
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime as dt
import seaborn as sns
import ensemble_tools as ens
import sql_tools as sq
from model_selection import mlin_regression, linear_map, mae, mape, rmse

PI_T_sup_dict = {'rundhoej':'4.141.120.29.HA.101', 'holme':'4.142.120.29.HA.101', 'hoerning':'4.146.120.29.HA.101'}
PI_T_ret_dict = {'rundhoej':'4.141.120.30.HA.101', 'holme':'4.142.120.30.HA.101', 'hoerning':'4.146.120.30.HA.101'}

PI_Q_dict = {'rundhoej':'H.141.181.02.HA.101', 'holme':'H.142.181.06.HA.101', 'hoerning':'K.146A.181.02.HA.101'}
PI_Q_dict2 = {'holme':'H.142.181.04.HA.101'}

Q_pump_max_dict = {'rundhoej':120, 'holme':430, 'hoerning':420}
Q_max_dict = {k:0.88*Q_pump_max_dict[k] for k in Q_pump_max_dict.keys()}

specific_heat_water = 1.17e-6 # MWh/kgKelvin
density_water = 980 # kg/m3 at 65 deg C
T_ret = 36.5
T_grnd = 5.9 # deg C (average ground temperature in 100 cm depth)

figpath = 'figures/heat_exchanger_model/'
# unicode charecters
uni_tothethird = u'\u00B3'
uni_degree = u'\u00B0'
uni_squared = u'\u00B2'



def gen_synthetic_cons(ens_preds, cons_pred, model_std):
    rand_ens_pred = pd.Series(index=ens_preds.index)
    rand_columns = np.random.randint(25,size=len(ens_preds))
    for i, c in zip(ens_preds.index, rand_columns):
        rand_ens_pred[i] = ens_preds.ix[i,c]
        
    weather_error = cons_pred - rand_ens_pred
    model_error = pd.Series(model_std*np.random.randn(len(cons_pred)), index=cons_pred.index)
    
    synth_cons = cons_pred - (weather_error + model_error)
    
    return synth_cons
    
    
def sim_with_synth_cons(model_std, ens_preds, cons_pred, sim_input):

    
    return synth_resid, synth_resid, sc2_errormargin_synth
    
#%%
    
def chi2_from_sig_m(sig_m, err_t, sig_w):
    sig_t = np.sqrt(sig_m**2 + sig_w**2)
    mean_err_t = np.mean(err_t)
    err_t_normalized = (err_t-mean_err_t)/sig_t
    vals, bins = np.histogram(err_t_normalized, bins='sturges')
    std_norm = stats.norm(loc=0, scale=1)
    
    normal_vals = [len(err_t)*integrate.quad(std_norm.pdf, bins[i], bins[i+1])[0] for i in range(len(vals))]
    print stats.chisquare(vals, normal_vals)[0]
    print sig_t.mean()
    print sum(vals), sum(normal_vals)
    return stats.chisquare(vals, normal_vals)[0]


def quant_from_nosigma(no_sigma):
    norm_dist = stats.norm(loc=0, scale=1)
    
    return norm_dist.cdf(no_sigma)
    
    
def nosigma_from_quant(quantile):
    norm_dist = stats.norm(loc=0, scale=1)
    
    return norm_dist.ppf(quantile)
    

def percent_above_forecasterrormargin(error_margin, forecast, actual):
    forecast_w_margin = forecast + error_margin
    no_aboves = np.count_nonzero(actual > forecast_w_margin)
    print float(no_aboves)
    
    return float(no_aboves)/len(actual)
    
def model_based_uncertainty_alaGorm(ens_preds, mean_forecast, actual, no_sigma, quantile):
    weather_uncert = no_sigma*ens_preds.std(axis=1)
    max_x = np.max(np.abs(mean_forecast-actual))
    xs = np.linspace(0, max_x, 1000)
    erf = np.array([np.abs((1.-quantile) - percent_above_forecasterrormargin(weather_uncert+x, mean_forecast, actual)) for x in xs])
    model_uncert = xs[np.argmin(erf)]
    #fmin(lambda x: np.abs((1.-quantile) - percent_above_forecasterrormargin(weather_uncert+x, mean_forecast, actual)), weather_uncert.mean(), ftol=0.00001)
    print "qua:", quantile
    
    plt.figure()
    plt.plot(np.linspace(0, max_x, 1000), erf,'-')
    plt.title('erf')
    return model_uncert
    
    
def model_based_sigma_alaChi2(ens_preds, mean_forecast, actual):
    error_total = mean_forecast - actual
    sigma_weather = ens_preds.std(axis=1)
    
    sigma_m = fmin(lambda sig_m:chi2_from_sig_m(sig_m, error_total, sigma_weather), sigma_weather.mean())[0]
    
    return sigma_m
    
def total_uncertainty_scale_alaChi2(ens_preds, mean_forecast, actual, quantile):
    sig_m = model_based_sigma_alaChi2(ens_preds, mean_forecast, actual)    
    sig_t = np.sqrt(ens_preds.std(axis=1)**2+sig_m**2)
    
    scale = fmin(lambda a: np.abs((1-quantile) - percent_above_forecasterrormargin(a*sig_t, mean_forecast, actual)), 1.)
    print "qua:", quantile
    return scale


def get_Tmin_func(df, T_min_q = 0.01, N_Q_q=11):
    
    Q_quantiles = [df['Q'].quantile(i) for i in np.linspace(0,1,N_Q_q)]
    Q_q_midpoints = Q_quantiles[:-1] + 0.5*np.diff(Q_quantiles)
    
    T_min_vals = [df[(df['Q']>Q_quantiles[i]) & (df['Q']<Q_quantiles[i+1])]['T_sup'].quantile(T_min_q) for i in range(N_Q_q-1)]       
    Tmin_func = interp1d(Q_q_midpoints, T_min_vals, kind='linear', fill_value="extrapolate")
        
    return Tmin_func, Q_quantiles

def get_TminofTout_func(df, station, frac_below = 0.005):
    slope = -0.74
    Tmin_dict = {'holme':65., 'rundhoej':64., 'hoerning':62.}
    Tmin = Tmin_dict[station]
     
    Tout_low_pass = df['Toutsmooth'] 
    def Tsuplim(Tout, b):
        return max(Tmin, slope*Tout+b)
        
    def no_points_below(b):
        res = []
        for To, Ts in zip(Tout_low_pass, df['T_sup']):
            if Ts < Tsuplim(To, b):
                res.append(1)
        return len(res)
           
    correct_b = fmin(lambda b:np.abs(no_points_below(b)-frac_below*len(df)), 70, disp=False)[0]
    
    return lambda T:Tsuplim(T, correct_b)
    

def TminofQref(TminofTout_func, Toutsmooth, Qref, P, T_ret):
    Toutmin = TminofTout_func(Toutsmooth)
    flowmin = T_ret + P/(specific_heat_water*density_water*Qref)

    return max(Toutmin, flowmin)


def Tsup_vs_Tout(df, station):      
    plt.figure()
    plt.scatter(df['Toutsmooth'], df['T_sup'])

    Toutforline = np.linspace(-10, 25, 100)
    TminofTout = get_TminofTout_func(df, station)
    Tsuplimforline = [TminofTout(T) for T in Toutforline]
    plt.plot(Toutforline, Tsuplimforline)
    plt.xlabel('Smoothed outside temperature') 
    plt.ylabel('Supply temperature')
    
    return
    
    
def QofPQref(TminofTout_func, Toutsmooth, Qref, P, T_ret):
    Q = fmin(lambda Q:np.abs(P-density_water*specific_heat_water*Q*(\
                TminofQref(TminofTout_func, Toutsmooth, Qref, P, T_ret) - T_ret)), Qref, disp=False)[0]
    
    return Q
    
def Qref_objfun(Qref, Q_max, TminofTout_func, Toutsmooth, P, deltaP, T_ret):
    return np.abs(Q_max - (P+deltaP)/(specific_heat_water*density_water\
                                     *(TminofQref(TminofTout_func, Toutsmooth, Qref, P, T_ret)-T_ret)))

    
def Qref_from_uncert(Q_max, TminofTout_func, Toutsmooth, P, deltaP, T_ret):
    Qref = fmin(lambda Qref:Qref_objfun(Qref, Q_max, TminofTout_func, Toutsmooth, P, deltaP, T_ret),\
                                                 0.8*Q_max, disp=False)[0]
    
    return Qref

def simulate_operation(sim_input, errormargin, TminofTout_fun, station):
    sim_results = pd.DataFrame(columns=['Q_ref', 'Q', 'T_sup', 'T_ret', 'cons'], index=sim_input.index)
    
    for ts in sim_input.index:      
        sim_results.loc[ts, 'Q_ref'] = Qref_from_uncert(Q_pump_max_dict[station], \
                                            TminofTout_fun, sim_input.loc[ts,'Toutsmooth'],\
                                            sim_input.loc[ts, 'cons_pred'], errormargin[ts], \
                                            sim_input.loc[ts, 'T_ret1hbefore'])
        sim_results.loc[ts, 'T_sup'] = TminofQref(TminofTout_fun, sim_input.loc[ts, 'Toutsmooth'], \
                                            sim_results.loc[ts, 'Q_ref'], sim_input.loc[ts, 'cons_pred'], \
                                            sim_input.loc[ts, 'T_ret1hbefore'])

        sim_results.loc[ts, 'Q'] = sim_input.loc[ts, 'cons']/(specific_heat_water*density_water*(sim_results.loc[ts, 'T_sup']-sim_input.loc[ts, 'T_ret']))
        sim_results.loc[ts, 'T_ret'] = sim_input.loc[ts, 'T_ret']
        sim_results.loc[ts, 'cons'] = specific_heat_water*density_water*sim_results.loc[ts, 'Q']*(sim_results.loc[ts, 'T_sup']-sim_results.loc[ts, 'T_ret'])

    return sim_results
    

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


def load_cons_model_dfs(df):
    # Takes the data frame with the already calculated consumptions
        #%%
    fit_ts = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    vali_ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    test_ts = ens.gen_hourly_timesteps(dt.datetime(2016,2,5,1), dt.datetime(2016,3,1,0))
    
    weathervars=['Tout', 'vWind', 'sunRad', 'hum']
    
    fit_data = pd.DataFrame()
    vali_data = pd.DataFrame()            
    test_data = pd.DataFrame()
    
    fit_data['cons'] = np.array(df.ix[fit_ts[0]:fit_ts[-1]]['cons'])
    vali_data['cons'] = np.array(df.ix[vali_ts[0]:vali_ts[-1]]['cons']) # the casting is a hack to avoid the index fucking up
    test_data['cons'] = np.array(df.ix[test_ts[0]:test_ts[-1]]['cons'])

    fit_data['cons24hbefore'] = np.array(df.ix[fit_ts[0]+dt.timedelta(days=-1):fit_ts[-1]+dt.timedelta(days=-1)]['cons']) 
    vali_data['cons24hbefore'] = np.array(df.ix[vali_ts[0]+dt.timedelta(days=-1):vali_ts[-1]+dt.timedelta(days=-1)]['cons']) # the casting is a hack to avoid the index fucking up
    test_data['cons24hbefore'] = np.array(df.ix[test_ts[0]+dt.timedelta(days=-1):test_ts[-1]+dt.timedelta(days=-1)]['cons'])

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
    
    for d, t in zip([fit_data, vali_data, test_data], [fit_ts, vali_ts, test_ts]):
        d.set_index(pd.DatetimeIndex(t), inplace=True)
                                                                   
    all_data = pd.concat([fit_data, vali_data, test_data])

    return fit_data, vali_data, test_data, all_data


def load_cons_model_ens_dfs(df):
    fit_ts = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
    vali_ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
    test_ts = ens.gen_hourly_timesteps(dt.datetime(2016,2,5,1), dt.datetime(2016,3,1,0))

    
    weathervars=['Tout', 'vWind', 'sunRad', 'hum']
    
    fit_data = [pd.DataFrame() for i in range(25)]
    vali_data = [pd.DataFrame() for i in range(25)]          
    test_data = [pd.DataFrame() for i in range(25)]
    
    for i in range(25):
        fit_data[i]['cons'] = np.array(df.ix[fit_ts[0]:fit_ts[-1]]['cons'])
        vali_data[i]['cons'] = np.array(df.ix[vali_ts[0]:vali_ts[-1]]['cons']) # the casting is a hack to avoid the index fucking up
        test_data[i]['cons'] = np.array(df.ix[test_ts[0]:test_ts[-1]]['cons'])

        fit_data[i]['cons24hbefore'] = np.array(df.ix[fit_ts[0]+dt.timedelta(days=-1):fit_ts[-1]+dt.timedelta(days=-1)]['cons']) 
        vali_data[i]['cons24hbefore'] = np.array(df.ix[vali_ts[0]+dt.timedelta(days=-1):vali_ts[-1]+dt.timedelta(days=-1)]['cons']) # the casting is a hack to avoid the index fucking up
        test_data[i]['cons24hbefore'] = np.array(df.ix[test_ts[0]+dt.timedelta(days=-1):test_ts[-1]+dt.timedelta(days=-1)]['cons'])

    for v in weathervars:
        all_ens_fit = ens.load_ens_timeseries_as_df(\
                                    ts_start=fit_ts[0],\
                                    ts_end=fit_ts[-1], \
                                    weathervars=[v]) \
                                  - ens.load_ens_timeseries_as_df(\
                                    ts_start=fit_ts[0]+dt.timedelta(days=-1),\
                                    ts_end=fit_ts[-1]+dt.timedelta(days=-1), \
                                    weathervars=[v])
        all_ens_vali = ens.load_ens_timeseries_as_df(\
                                    ts_start=vali_ts[0],\
                                    ts_end=vali_ts[-1], \
                                    weathervars=[v]) \
                                  - ens.load_ens_timeseries_as_df(\
                                    ts_start=vali_ts[0]+dt.timedelta(days=-1),\
                                    ts_end=vali_ts[-1]+dt.timedelta(days=-1), \
                                    weathervars=[v])
        all_ens_test = ens.load_ens_timeseries_as_df(\
                                    ts_start=test_ts[0],\
                                    ts_end=test_ts[-1], \
                                    weathervars=[v]) \
                                  - ens.load_ens_timeseries_as_df(\
                                    ts_start=test_ts[0]+dt.timedelta(days=-1),\
                                    ts_end=test_ts[-1]+dt.timedelta(days=-1), \
                                    weathervars=[v])
                                    
        for i in range(25):
            fit_data[i]['%s24hdiff'%v] = all_ens_fit[v + str(i)]
            vali_data[i]['%s24hdiff'%v] = all_ens_vali[v + str(i)]
            test_data[i]['%s24hdiff'%v] = all_ens_test[v + str(i)]
    
    all_data = []
    for i in range(25):
        for d, t in zip([fit_data[i], vali_data[i], test_data[i]], [fit_ts, vali_ts, test_ts]):
            d.set_index(pd.DatetimeIndex(t), inplace=True)
        all_data.append(pd.concat([fit_data[i], vali_data[i], test_data[i]]))
        
    return all_data

def fig_error_margins(sc2_errormargin, sc3_errormargin, sim_input, sc3_model_uncert, station, no_sigma):
    plt.figure()
    plt.plot_date(sim_input.index, sc2_errormargin, 'r-', label='Scenario 2')
    plt.plot_date(sim_input.index, sc3_errormargin, 'g-', label='Scenario 3')
    plt.plot_date(sim_input.index, sim_input['cons']-sim_input['cons_pred'], 'y-', label='Forecast_error')
    plt.title("Errormargin on forecast: " + station + ', ' + str(no_sigma) + r'$\sigma$' + ' Model uncert: ' +str(sc3_model_uncert))
    plt.ylabel("[MW]")
    plt.legend()
    
    return
    
def fig_heat_loss(sim_input, sim_results_sc2, sim_results_sc3, station, no_sigma, figfilename=None, figpath=figpath, save=True):
    fig, axes = plt.subplots(2,2, figsize=(30,7.5), sharex=True)
    axes[0,0].plot_date(sim_input.index, sim_input['T_sup'] - sim_results_sc2['T_sup'], 'k-', label='Absolute reduction in supply temperature')
    axes[0,0].set_title('Scenario 1 vs Scenario 2')
    axes[0,0].annotate('mean: %2.3f' % (np.mean(sim_input['T_sup'] - sim_results_sc2['T_sup'])), xy=(0.05, 0.1), xycoords='axes fraction')
    axes[0,0].set_ylabel(u'Differens in supply temperature 1-2 [%sC]'%uni_degree, size=8)
    axes[1,0].plot_date(sim_input.index, 100*(1-(sim_results_sc2['T_sup']-T_grnd)/(sim_input['T_sup'] - T_grnd)), '-')
    axes[1,0].set_ylabel('Heat loss\nreduction [%]', size=8)    
    axes[1,0].set_title('Scenario 1 --> Scenario 2')
    axes[1,0].annotate('mean: %2.3f' % (np.mean(100*(1-(sim_results_sc2['T_sup']-T_grnd)/(sim_input['T_sup'] - T_grnd)))), xy=(0.05, 0.1), xycoords='axes fraction')

    axes[0,1].plot_date(sim_input.index, sim_results_sc2['T_sup'] - sim_results_sc3['T_sup'], 'k-', label='Absolute reduction in supply temperature')
    axes[0,1].set_title('Scenario 2 vs Scenario 3')
    axes[0,1].annotate('mean: %2.3f' % (np.mean(sim_results_sc2['T_sup'] - sim_results_sc3['T_sup'])), xy=(0.05, 0.1), xycoords='axes fraction')

    axes[0,1].set_ylabel(u'Differens in supply temperature 2-3 [%sC]'%uni_degree, size=8)    
    axes[1,1].plot_date(sim_input.index, 100*(1-(sim_results_sc3['T_sup']-T_grnd)/(sim_results_sc2['T_sup'] - T_grnd)), '-')
    axes[1,1].set_ylabel('Heat loss\nreduction [%]', size=8) 
    axes[1,1].set_title('Scenario 2 --> Scenario 3')
    axes[1,1].annotate('mean: %2.3f' % (np.mean(100*(1-(sim_results_sc3['T_sup']-T_grnd)/(sim_results_sc2['T_sup'] - T_grnd)))), xy=(0.05, 0.1), xycoords='axes fraction')

    
    fig.suptitle(station + ', ' + str(no_sigma) + r'$\sigma$')
    if figfilename==None:
        figfilename = 'heat_loss_%2.2f'%(no_sigma) + 'sigma_' + station + '.pdf'
    if save:
        fig.savefig(figpath + figfilename)
    
    return

#%%
def main(argv):
    plt.close('all')
    
    try:
        station = argv[0]
        no_sigma = argv[1]
        if not station in PI_T_sup_dict.keys():
            print "Use rundhoej, holme or hoerning and a float for the uncertainty bound"
            return
    except:
        print "No station provided. Defaults to holme, no_sigma=2"
        station = 'holme'
        no_sigma=2
        
    print station, no_sigma
    # old tsstart dt.datetime(2014,12,17,1)
    ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,3,1,1), dt.datetime(2016,1,15,0))
    ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,19,1), dt.datetime(2016,3,1,0))
    all_ts = ts1 + ts2
       
    
    
    df = pd.DataFrame(index=all_ts)
    if station == 'holme':
        PI_Q1 = PI_Q_dict[station]
        PI_Q2 = PI_Q_dict2[station]
        df['Q1']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q1, ts1[0], ts1[-1]),\
                    sq.fetch_hourly_vals_from_PIno(PI_Q1, ts2[0], ts2[-1])])
        df['Q2']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q2, ts1[0], ts1[-1]),\
                    sq.fetch_hourly_vals_from_PIno(PI_Q2, ts2[0], ts2[-1])])
        df['Q'] = df['Q1']+df['Q2']    
    else:
        PI_Q = PI_Q_dict[station]
        df['Q']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q, ts1[0], ts1[-1]),\
                    sq.fetch_hourly_vals_from_PIno(PI_Q, ts2[0], ts2[-1])])
    
    PI_T_sup = PI_T_sup_dict[station]
    df['T_sup']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts1[0], \
                    ts1[-1]),sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts2[0], ts2[-1])])    
    PI_T_ret = PI_T_ret_dict[station]
    df['T_ret']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_T_ret, ts1[0], \
                    ts1[-1]),sq.fetch_hourly_vals_from_PIno(PI_T_ret, ts2[0], ts2[-1])]) 
    df['ts'] = all_ts
    df['cons'] = specific_heat_water*density_water*df['Q']*(df['T_sup']-df['T_ret'])
    Tout1 = sq.fetch_BrabrandSydWeather('Tout', ts1[0], ts1[-1])
    Tout2 = sq.fetch_BrabrandSydWeather('Tout', ts2[0], ts2[-1])
    Tout = np.concatenate([Tout1, Tout2])
    Tout_low_pass = [Tout[range(i-23,i+1)].mean() for i in range(len(Tout))]
    df['Toutsmooth'] = Tout_low_pass

    Tsup_vs_Tout(df, station)

   
    
    #%% fitting and testing consumption prediction
    fit_data, vali_data, test_data, all_data = load_cons_model_dfs(df)
    fit_y = fit_data['cons']
    columns = ['cons24hbefore', 'Tout24hdiff', 'vWind24hdiff', 'sunRad24hdiff']
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
    
    ens_dfs = load_cons_model_ens_dfs(df)
    ens_preds = np.empty((len(ens_dfs[0]), len(ens_dfs)))
    for edf, i in zip(ens_dfs, range(len(ens_dfs))):
        ens_pred = linear_map(edf, res.params, columns)
        ens_preds[:,i] = ens_pred
        plt.plot_date(all_data.index, ens_pred, 'grey', lw=0.5)
        
    ens_preds = pd.DataFrame(ens_preds, index=all_data.index)
    plt.plot_date(all_data.index, all_data['cons'], 'k-', lw=2)
    plt.plot_date(all_data.index, np.concatenate([res.fittedvalues, vali_pred, test_pred]), 'r-', lw=2)
    plt.title(station + ' forecasts of consumption')
    nonfit_errors = pd.concat([valierr, testerr])
    
    all_pred = np.concatenate([res.fittedvalues, vali_pred, test_pred])
    all_pred = pd.Series(all_pred, index=all_data.index)
    print res.summary()
    
    #%% 
    TminofTout_fun = get_TminofTout_func(df, station, frac_below = 0.005)    

    sim_input = df.ix[all_data.index]
    sim_input['T_ret1hbefore'] = np.roll(sim_input['T_ret'], 1)
    sim_input['cons_pred'] = all_pred
    
    
    
    sc2_errormargin = pd.Series(no_sigma*np.ones(len(sim_input))*nonfit_errors.std(), index=sim_input.index)
    
    nonfit_ts_start = vali_data.index[0]
    nonfit_ts_end = test_data.index[-1]
    
    quantile_sc2 = 1. - percent_above_forecasterrormargin(\
                    sc2_errormargin.loc[nonfit_ts_start:nonfit_ts_end], \
                    sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                    sim_input.loc[nonfit_ts_start:nonfit_ts_end,'cons'])
    sc3_model_uncert = model_based_uncertainty_alaGorm(\
                            ens_preds.loc[nonfit_ts_start:nonfit_ts_end], \
                            sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                            sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons'], no_sigma, quantile_sc2)    
    sc3_errormargin = pd.Series(no_sigma*ens_preds.std(axis=1) + sc3_model_uncert,  index=sim_input.index)

    sig_m = model_based_sigma_alaChi2(ens_preds.loc[nonfit_ts_start:nonfit_ts_end], sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons'])    
    sig_t = np.sqrt(ens_preds.std(axis=1)**2+sig_m**2)
    sc35scale = total_uncertainty_scale_alaChi2(\
                                ens_preds.loc[nonfit_ts_start:nonfit_ts_end],\
                                sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'],\
                                sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons'],\
                                quantile_sc2)    
    print sig_m    
    #sc35_errormargin = pd.Series(no_sigma*np.sqrt(ens_preds.std(axis=1)**2+sig_m**2), index=sim_input.index)
    sc35_errormargin = pd.Series(sc35scale*sig_t, index=sim_input.index)
    
        
    use_sc35 = False
    if use_sc35:
        sc3_errormargin = sc35_errormargin
     
    sim_results_sc2 = simulate_operation(sim_input, sc2_errormargin, TminofTout_fun, station)
    sim_results_sc3 = simulate_operation(sim_input, sc3_errormargin, TminofTout_fun, station)    
    
    #%% synthetic consumption, controlled variable model uncertainty
    
    model_stds = [0.5*sim_input['cons'].std(), 0.1*sim_input['cons'].std(), 0.05*sim_input['cons'].std()]# sim_input['cons'].std()*np.linspace(0,1,10)
    sc2_synth_results = []
    sc3_synth_results = []
    model_uncerts = []
    for model_std in model_stds:
        synth_cons = gen_synthetic_cons(ens_preds, sim_input['cons_pred'], model_std)
        sim_input_synth = sim_input.copy(deep=True)
        sim_input_synth['cons'] = synth_cons
        synth_resid = sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'] - sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end, 'cons']
        sc2_errormargin_synth = pd.Series(no_sigma*np.ones(len(sim_input_synth))*synth_resid.std(), index=sim_input_synth.index)
        quantile_sc2_synth = 1. - percent_above_forecasterrormargin(\
                        sc2_errormargin_synth.loc[nonfit_ts_start:nonfit_ts_end], \
                        sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                        sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end,'cons'])
        print "Sc2 q: ", quantile_sc2_synth
        sc3_model_uncert_synth = model_based_uncertainty_alaGorm(\
                                ens_preds.loc[nonfit_ts_start:nonfit_ts_end], \
                                sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                                sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end, 'cons'], no_sigma, quantile_sc2_synth)
        model_uncerts.append(sc3_model_uncert_synth)
        sc3_errormargin_synth = pd.Series(no_sigma*ens_preds.std(axis=1) + sc3_model_uncert_synth,  index=sim_input_synth.index)
    
        sim_results_sc2_synth = simulate_operation(sim_input_synth, sc2_errormargin_synth, TminofTout_fun, station)
        sim_results_sc3_synth = simulate_operation(sim_input_synth, sc3_errormargin_synth, TminofTout_fun, station)
        sc2_synth_results.append(sim_results_sc2_synth)
        sc3_synth_results.append(sim_results_sc3_synth)

    mean_Tsupdiff = []
    mean_heatlossreduced = []
    for sc2_res, sc3_res in zip(sc2_synth_results, sc3_synth_results):
        mean_Tsupdiff.append(np.mean(sc2_res['T_sup'] - sc3_res['T_sup']))
        mean_heatlossreduced.append(np.mean(100*(1-(sc3_res['T_sup']-T_grnd)/(sc2_res['T_sup'] - T_grnd))))
        
    plt.figure()
    plt.plot(model_uncerts, mean_Tsupdiff, 'k.')
    plt.title('Mean temp reduction vs model uncert.')
        
    print "Perc above errormargin, sc2: ", percent_above_forecasterrormargin(\
                    sc2_errormargin.loc[nonfit_ts_start:nonfit_ts_end], \
                    sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                    sim_input.loc[nonfit_ts_start:nonfit_ts_end,'cons'])
    print "Perc above errormargin, sc3: ", percent_above_forecasterrormargin(sc3_errormargin.loc[nonfit_ts_start:nonfit_ts_end], \
                    sim_input.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                    sim_input.loc[nonfit_ts_start:nonfit_ts_end,'cons'])
    print "mean errormargin, sc2: ", sc2_errormargin.mean()
    print "mean errormargin, sc3: ", sc3_errormargin.mean()
    print "rms errormargin, sc2: ", rmse(sc2_errormargin)
    print "rms errormargin, sc3: ", rmse(sc3_errormargin)
    
    print "Synth Perc above errormargin, sc2: ", percent_above_forecasterrormargin(\
                    sc2_errormargin_synth.loc[nonfit_ts_start:nonfit_ts_end], \
                    sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                    sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end,'cons'])
    print "Synth  Perc above errormargin, sc3: ", percent_above_forecasterrormargin(sc3_errormargin_synth.loc[nonfit_ts_start:nonfit_ts_end], \
                    sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end, 'cons_pred'], \
                    sim_input_synth.loc[nonfit_ts_start:nonfit_ts_end,'cons'])
    print "Synth mean errormargin, sc2: ", sc2_errormargin_synth.mean()
    print "Synth mean errormargin, sc3: ", sc3_errormargin_synth.mean()
    print "Synth rms errormargin, sc2: ", rmse(sc2_errormargin_synth)
    print "Synth rms errormargin, sc3: ", rmse(sc3_errormargin_synth)

    
    #% error margins:
    fig_error_margins(sc2_errormargin, sc3_errormargin, sim_input, sc3_model_uncert, station, no_sigma)
    fig_error_margins(sc2_errormargin_synth, sc3_errormargin_synth, sim_input_synth, sc3_model_uncert_synth, station, no_sigma)
    
    sns.jointplot(np.abs(nonfit_errors), ens_preds.loc[nonfit_ts_start:nonfit_ts_end].std(axis=1))
    sns.jointplot(np.abs(synth_resid), ens_preds.loc[nonfit_ts_start:nonfit_ts_end].std(axis=1))


    #% T Q scatter plots
    fig, axes = plt.subplots(3,1, figsize=(10,16), sharex=True, sharey=True)
    axes[0].scatter(sim_input['T_sup'], sim_input['Q'], c=sim_input['cons'])
    axes[0].set_title(station + ': ' + 'Scenario 1')
    
    axes[1].scatter(sim_results_sc2['T_sup'], sim_results_sc2['Q'], c=sim_results_sc2['cons'])
    axes[1].set_title(station + ': Scenario 2: ' + str(no_sigma) + r'$\sigma$' )
    axes[2].scatter(sim_results_sc3['T_sup'], sim_results_sc3['Q'], c=sim_results_sc3['cons'])
    axes[2].set_title(station + ': Scenario 3: ' + str(no_sigma) + r'$\sigma$')
    axes[1].set_ylabel(u'Water flow rate [m%s/h]'%uni_tothethird, size=8)
    axes[2].set_xlabel(u'Supply temperature [%sC]'%uni_degree, size=8)
    fig.tight_layout()
    fig.savefig(figpath + 'TQscatter_%2.2f'%(no_sigma)  + 'sigma_' + station + '.pdf')

    # T_sup time series fig
    fig, axes = plt.subplots(3,1, figsize=(15,15), sharex=True)
    axes[0].plot_date(sim_input.index, sim_input['T_sup'], 'k-', label='Scenario 1')
    axes[0].plot_date(sim_input.index, sim_results_sc2['T_sup'], 'r-', lw=3, label='Scenario 2')
    axes[0].plot_date(sim_input.index, sim_results_sc2['T_sup'], 'g-', label='Scenario 3')
    axes[0].set_title(station + ', ' + str(no_sigma) + r'$\sigma$' + ': Supply temperature')
    axes[0].set_ylabel(u'Supply temperature [%sC]'%uni_degree, size=8)    
    axes[0].legend()
    axes[1].plot_date(sim_input.index, sim_input['Q'], 'k-', label='Scenario 1' )
    axes[1].plot_date(sim_input.index, sim_results_sc2['Q'], 'r-', label='Scenario 2')
    axes[1].plot_date(sim_input.index, sim_results_sc2['Q_ref'], 'b-', lw=1, label=r'$Q_{ref}$' + 'Scenario 2')
    axes[1].set_ylabel(u'Water flow rate [m%s/h]'%uni_tothethird, size=8)
    axes[1].legend()
    axes[2].plot_date(sim_input.index, sim_input['Q'], 'k-', label='Scenario 1' )
    axes[2].plot_date(sim_input.index, sim_results_sc3['Q'], 'g-', label='Scenario 3')
    axes[2].plot_date(sim_input.index, sim_results_sc3['Q_ref'], 'b-', lw=1, label=r'$Q_{ref}$' + 'Scenario 3')
    axes[2].set_ylabel(u'Water flow rate [m%s/h]'%uni_tothethird, size=8)
    axes[2].legend()
    fig.savefig(figpath + 'TQtimeseries_%2.2f'%(no_sigma) + 'sigma_' + station + '.pdf')
    
    # Differencen in supply temperature between the scenarios
    fig_heat_loss(sim_input, sim_results_sc2, sim_results_sc3, station, no_sigma)
    fig_heat_loss(sim_input_synth, sim_results_sc2_synth, sim_results_sc3_synth, station, no_sigma, save=False)
        
    
    return 
    
    #%% The below section only runs if we view Tmin as a function of Q (the old way)
    # note: SOME OF THIS USES CONSTANT TRET!!
    TminofQ = False
    if TminofQ:    
        # outlierdetection
        X = df[['T_sup','Q']]
        outlier_detection = False
        if outlier_detection: 
            detect_outliers(X, station)
        else:
            inlierpred = np.ones(len(df), dtype=bool)
              
    
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        cond_df = df
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

   
   
def try_different_uncerts():
    for station in ['rundhoej', 'holme', 'hoerning']:
        for no_sigma in [nosigma_from_quant(0.95), nosigma_from_quant(0.975), 2., nosigma_from_quant(0.99), 3., nosigma_from_quant(0.999)]:
            main([station, no_sigma])
            
    return
    
if __name__ == "__main__":
    main(sys.argv[1:])