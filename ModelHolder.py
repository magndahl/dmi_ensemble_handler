import pandas as pd
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
        self.predict_X = None
        self.predict_y = None


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



def gen_SVR_fit_data(ts_start, ts_end):
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
    most_recent_avail_prod = sq.fetch_production(h_hoursbefore(ts_start, 24),\
                                                          h_hoursbefore(ts_end, 24))
    ts = ens.gen_hourly_timesteps(ts_start, ts_end)
    for i, t, p48 in zip(range(len(most_recent_avail_prod)), ts, X['prod48hbefore']):
        if t.hour > 8 or t.hour == 0:
            most_recent_avail_prod[i] = p48

    X['prod24or48hbefore'] = most_recent_avail_prod
    y = sq.fetch_production(ts_start, ts_end)

    return X, y


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

