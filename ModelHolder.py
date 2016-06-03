import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
import sql_tools as sq
import ensemble_tools as ens

class ModelHolder(object):
    """ This class is intended as for holding a model,
        of like LinearRegression or SVR from the sklearn
        module, along with methods to prepare data to fit
        the model and data to predict based on the model.

        """

    def __init__(self, model, gen_fit_data_func, gen_predict_data_func):
        """ model must be have a .fit(X)-method like the models
            from sklearn.
            gen_fit_data_func has to return X, y
            of data to fit.
            gen_predict_data_func must return a dictionary
            of predictor data-sets for the period of interest.


            """

        self.model = model
        self.gen_fit_data_func = gen_fit_data_func
        self.gen_predict_data_func = gen_predict_data_func

        self.fit_X = None
        self.fit_y = None
        self.X_scaler = None
        self.fit_X_scaled = None
        self.y_scaler = None
        self.fit_y_scaled = None
        self.predict_X_dict = None
        self.predict_y_dict = None


    def gen_fit_data(self, *kwargs):
        """ This method calls the gen_fit_data_func function
            and sets the fit_X and fit_y field to the result.

            """

        self.fit_X, self.fit_y = self.gen_fit_data_func(*kwargs)


    def scale_fit_vars(self):
        """ This method can only be called after gen_fit_data
            and creates 4 new fields:
            fit_X_scaled, fit_y_scaled, X_scaler and y_scaler
            The scalers are sklearn StandardScaler's

            """

        self.X_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(self.fit_X)
        self.fit_X_scaled = self.X_scaler.transform(self.fit_X)
        self.y_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(self.fit_y.reshape(-1,1))
        self.fit_y_scaled = self.y_scaler.transform(self.fit_y.reshape(-1,1)).reshape(-1,)


    def gen_predict_data(self, *kwargs):
        """ This method calls the gen_fit_data_func function
            and sets the predict_X_dict field to the result.

            """

        self.predict_X_dict = self.gen_predict_data_func(*kwargs)



    def fit(self):
        """ This method fits the model to scaled data if
            possible. If fit data has not been scaled,
            it fits to the unscaled, but prints a warning.

            """

        try:
            self.model.fit(self.fit_X_scaled, self.fit_y_scaled)
        except:
            print "No scaled vars, uses fits to uncaled!"
            self.model.fit(self.fit_X, self.fit_y)


    def predict(self):
        if self.X_scaler!=None:
            self.predict_y_dict = {}
            for k in self.predict_X_dict.keys():
                self.predict_y_dict[k] = self.y_scaler.inverse_transform(\
                                        self.model.predict(self.X_scaler.transform(\
                                                       self.predict_X_dict[k])))
        else:
            self.predict_y_dict = {}
            for k in self.predict_X_dict.keys():
                self.predict_y_dict[k] = self.model.predict(self.predict_X_dict[k])




def gen_SVR_ens_mean_X(ts_start, ts_end):
    """ This function can generate fit data for an SVR
        model based on 20 variables.

        """

    varnames=['Tout', 'vWind', 'hum', 'sunRad']
    X = gen_lagged_w_ens_mean_diff_df(ts_start, ts_end, \
                                    varnames=varnames,\
                                    timeshifts = [48, 60, 168])
    for v in varnames:
        #include absolute weather vars
        X[v] = ens.load_ens_mean_avail_at10_series(v, ts_start, ts_end)

    # include the most recent avail prod at 9.45, that is the one from 8o'clock
    last_avail_hour = 8 # this must be changed if the horizon is changed
    most_recent_avail_prod = np.hstack([sq.fetch_production(h_hoursbefore(ts_start, 24),\
                                                          h_hoursbefore(ts_start+\
                                                          dt.timedelta(hours=last_avail_hour-1), 24)),\
                                       sq.fetch_production(h_hoursbefore(ts_start+\
                                                          dt.timedelta(hours=last_avail_hour), 48),\
                                                          h_hoursbefore(ts_end, 48))])

    X['prod24or48hbefore'] = most_recent_avail_prod

    return X


def gen_SVR_fit_data(ts_start, ts_end):
    X = gen_SVR_ens_mean_X(ts_start, ts_end)
    y = sq.fetch_production(ts_start, ts_end)

    return X, y


def gen_SVR_predict_data(ts_start, ts_end):
    """ timeshifts must be integer number of hours. Posetive values only,
        dataframe contains columns with the variables minus their value
        'timeshift' hours before. """

    varnames = ['Tout', 'vWind', 'hum', 'sunRad']
    timeshifts = [48, 60, 168]

    df = pd.DataFrame()
    X_dict = {'ens_mean':gen_SVR_ens_mean_X(ts_start, ts_end)}
    df_s = [pd.DataFrame() for i in range(25)]

    for timeshift in timeshifts:

        prod_before = sq.fetch_production(h_hoursbefore(ts_start, timeshift),\
                                                          h_hoursbefore(ts_end, timeshift))
        for df in df_s:
            df['prod%ihbefore'%timeshift] = prod_before

        for v in varnames:
            ens_data = ens.load_ens_avail_at10_series(ts_start, ts_end, v, pointcode=71699)
            ens_data_before = ens.load_ens_avail_at10_series(h_hoursbefore(ts_start, timeshift),\
                                                        h_hoursbefore(ts_end, timeshift), v, pointcode=71699)
            diff = ens_data - ens_data_before
            for i in range(ens_data.shape[1]):
                df_s[i]['%s%ihdiff%i'%(v,timeshift, i)] = diff[:,i]
    for v in varnames:
        ens_data = ens.load_ens_avail_at10_series(ts_start, ts_end, v, pointcode=71699)
        for i in range(ens_data.shape[1]):
            df_s[i]['%s%i'%(v, i)] = ens_data[:,i]

    for df, i in zip(df_s, range(len(df_s))):
        df['prod24or48hbefore'] = X_dict['ens_mean']['prod24or48hbefore']
        X_dict['ens_%i' % i] = df

    return X_dict


def gen_lagged_w_ens_mean_diff_df(ts_start, ts_end, varnames, timeshifts, pointcode=71699):
    """ This function creates a dataframe of time lagged production time
        series as well as differenced weather variables corresponding to
        the time lags.
        timeshifts must be integer number of hours. Posetive values only,
        dataframe contains columns with the variables minus their value
        'timeshift' hours before.

        """

    df = pd.DataFrame()

    for timeshift in timeshifts:
        df['prod%ihbefore'%timeshift] = sq.fetch_production(h_hoursbefore(ts_start, timeshift),\
                                                          h_hoursbefore(ts_end, timeshift))
        for v in varnames:
            ens_mean = ens.load_ens_mean_avail_at10_series(v, ts_start, ts_end, pointcode=71699)
            ens_mean_before = ens.load_ens_mean_avail_at10_series(v,\
                                            h_hoursbefore(ts_start, timeshift),\
                                            h_hoursbefore(ts_end, timeshift),\
                                            pointcode=71699)
            df['%s%ihdiff'%(v,timeshift)] = ens_mean - ens_mean_before

    return df


def h_hoursbefore(timestamp, h):
    return timestamp + dt.timedelta(hours=-h)

