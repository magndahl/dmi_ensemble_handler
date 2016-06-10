#!/home/magnus/anaconda2/bin/python

import sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from scipy.stats import norm, chisquare
from scipy import integrate
from scipy.optimize import fmin
import ModelHolder as mh
import sql_tools as sq
from error_tools import rmse


def main(argv):
    ts_first = dt.datetime(2016,2,26,9)


    nows = [ts_first + dt.timedelta(days=i) for i in range(21)]

    predictions = []
    timesteps = []
    for ts_now in nows:
        print ts_now
        ts_today_h1 = dt.datetime(ts_now.year, ts_now.month, ts_now.day, 1)
        ts_fit_start = mh.h_hoursbefore(ts_today_h1, 31*24)
        ts_fit_end = mh.h_hoursbefore(ts_today_h1, 1)
        ts_predict_start = ts_today_h1 + dt.timedelta(hours=24)
        ts_predict_end = ts_today_h1 + dt.timedelta(hours=2*24-1)

        my_mh = mh.ModelHolder(model=SVR(kernel='rbf', C=2, gamma=0.003, epsilon=0.05),
                                    gen_fit_data_func=mh.gen_SVR_fit_data,
                                    gen_predict_data_func=mh.gen_SVR_predict_data)

        my_mh.gen_fit_data(ts_fit_start, ts_fit_end)
        my_mh.scale_fit_vars()
        my_mh.fit()

        my_mh.gen_predict_data(ts_predict_start, ts_predict_end)
        my_mh.predict()

        predictions.append(my_mh.predict_y_dict)
        timesteps.append(pd.date_range(ts_predict_start, ts_predict_end, freq='H'))

    mean_pred = np.hstack([p['ens_mean'] for p in predictions])
    timesteps_arr = np.hstack(timesteps)

    ens_pred = np.vstack([np.hstack([p['ens_%i'%i] for p in predictions]) for i in range(25)]).transpose()

    sig_w = ens_pred.std(axis=1)
    prod = sq.load_local_production(timesteps_arr[0], timesteps_arr[-1])

    err_t = mean_pred - prod
    sig_m = fmin(chi2_from_siq_m, 40, args=(err_t,sig_w))[0]

    sig_t = np.sqrt(sig_m**2+sig_w**2)
    print sig_t
    print sig_t.mean()
    print rmse(err_t)
    print err_t.std()
    print err_t.mean()
    print sig_w.mean()
    print sig_m



def chi2_from_siq_m(sig_m, err_t, sig_w):
    sig_t = np.sqrt(sig_m**2 + sig_w**2)
    mean_err_t = np.mean(err_t)
    err_t_normalized = (err_t-mean_err_t)/sig_t
    vals, bins = np.histogram(err_t_normalized, bins='sturges')
    std_norm = norm(loc=0, scale=1)
    normal_vals = [len(err_t)*integrate.quad(std_norm.pdf, bins[i], bins[i+1])[0] for i in range(len(vals))]

    return chisquare(vals, normal_vals)[0]


if __name__ == "__main__":
    main(sys.argv[1:])
