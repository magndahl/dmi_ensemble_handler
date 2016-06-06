#!/home/magnus/anaconda2/bin/python

import datetime as dt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import ModelHolder as mh


ts_first = dt.datetime(2016,2,26,9)


nows = [ts_first + dt.timedelta(days=i) for i in range(100)]

predictions = []
timesteps = []
for ts_now in nows:
    print ts_now
    ts_today_h1 = dt.datetime(ts_now.year, ts_now.month, ts_now.day, 1)
    ts_fit_start = mh.h_hoursbefore(ts_today_h1, 31*24)
    ts_fit_end = mh.h_hoursbefore(ts_today_h1, 1)
    ts_predict_start = ts_today_h1 + dt.timedelta(hours=24)
    ts_predict_end = ts_today_h1 + dt.timedelta(hours=2*24-1)

    my_mh = mh.ModelHolder(model=SVR(kernel='rbf', C=15, gamma=0.00266, epsilon=0.05),
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
