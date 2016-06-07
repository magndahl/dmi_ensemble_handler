#!/home/magnus/anaconda2/bin/python

import datetime as dt
import numpy as np
from sklearn.svm import SVR
import ModelHolder as mh

my_mh = mh.ModelHolder(model=SVR(kernel='rbf', C=35, gamma=0.00266),
                                gen_fit_data_func=mh.gen_SVR_fit_data,
                                gen_predict_data_func=mh.gen_SVR_predict_data)


print str(my_mh.model)

ts1f = dt.datetime(2016,2,29,1)
ts2f = dt.datetime(2016,3,29,0)
ts1p = dt.datetime(2016,3,30,1)

ts2p = dt.datetime(2016,3,31,0)
my_mh.gen_fit_data(ts1f, ts2f)
my_mh.scale_fit_vars()
my_mh.fit()

u = my_mh.model.predict(my_mh.fit_X_scaled)
print u
print my_mh.X_scaler

my_mh.gen_predict_data(ts1p, ts2p)
my_mh.predict()
print my_mh.predict_y_dict

