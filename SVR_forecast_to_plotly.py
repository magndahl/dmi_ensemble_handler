#!/home/magnus/anaconda2/bin/python

import sys, getopt
import sql_tools as sq
import ensemble_tools as ens
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg") # this is needed to rund under cron
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import plotly.plotly as ply
import plotly.tools as tls
import plotly.graph_objs as go
from sklearn.svm import SVR
from error_tools import rmse, mae, mape
import ModelHolder as mh


def main(argv):
    if len(argv)==0:
        print "No argument given, defaults to online mode"
        today = dt.datetime.now().date()
    elif argv[0]=='online':
        today = dt.datetime.now().date()
    else:
        try:
            today = dt.datetime.strptime(argv[0], '%Y-%m-%d').date()
        except ValueError:
            print "Argument must be 'online' or a valid date in yyyy-mm-dd format."
            sys.exit()

    print "Today is: %s" % today


    plyfig_yest = plot_yesterday_forcast(today=today)
    ply.plot(plyfig_yest, filename="Yesterday's SVR forecast", auto_open=False)

    plyfig_tom = plot_tomorrow_forecast(today=today)
    ply.plot(plyfig_tom, filename="SVR forecast for tomorrow", auto_open=False)

    return plyfig_yest


def plot_tomorrow_forecast(today):
    fig = plt.figure()
    fc_dict = forecast_tomorrow(today=today)
    mean_forecast = fc_dict['ens_mean']
    ens_forecast = np.array([fc_dict['ens_%i'%i] for i in range(25)]).transpose()
    ts = get_tomorrow_timesteps(today)
    plt.plot_date(ts, ens_forecast, '-', color='0.25', lw=0.3)
    plt.plot_date(ts, mean_forecast, 'b-', lw=2, label='Forecast')
    plt.ylabel("Production [MW]")
    plt.title("Tomorrow: " + str(today+dt.timedelta(days=1)))

    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig['layout']['showlegend'] = True
    plotly_fig['layout']['legend'] = dict(x=0.75, y=0.05)
    for d in plotly_fig.data:
        if d['name'][0:5]=='_line':
            d['showlegend'] = False

    return plotly_fig


def plot_yesterday_forcast(today):
    fig = plt.figure()
    day_before_yest = mh.h_hoursbefore(today, 48)
    fc_dict = forecast_tomorrow(day_before_yest)
    mean_forecast = fc_dict['ens_mean']
    ens_forecast = np.array([fc_dict['ens_%i'%i] for i in range(25)]).transpose()
    ts = get_tomorrow_timesteps(day_before_yest)

    prod = sq.load_local_production(ts[0], ts[-1])
    plt.plot_date(ts, prod, 'k-', lw=2, label='Realized production')
    plt.plot_date(ts, ens_forecast, '-', color='0.25', lw=0.3)
    plt.plot_date(ts, mean_forecast, 'b-', lw=2, label='Forecast')
    plt.ylabel("Production [MW]")
    plt.title("Yesterday: " + str(today+dt.timedelta(days=-1)))

    err = mean_forecast - prod
    err_string = """RMSE = %2.2f MW\nMAE = %2.2f MW\nMAPE = %2.2f %%""" \
                % (rmse(err), mae(err), 100*mape(err, prod))
    plt.annotate(err_string, xy=(0.80, 0.80), xycoords='axes fraction')

    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig['layout']['showlegend'] = True
    plotly_fig['layout']['legend'] = dict(x=0.75, y=0.05)
    for d in plotly_fig.data:
        if d['name'][0:5]=='_line':
            d['showlegend'] = False

    return plotly_fig



def forecast_tomorrow(today, C=2, gamma=0.003, epsilon=0.05):
    ts_today_h1 = dt.datetime(today.year, today.month, today.day, 1)
    ts_fit_start = mh.h_hoursbefore(ts_today_h1, 31*24)
    ts_fit_end = mh.h_hoursbefore(ts_today_h1, 1)
    ts_predict_start = ts_today_h1 + dt.timedelta(hours=24)
    ts_predict_end = ts_today_h1 + dt.timedelta(hours=2*24-1)

    my_mh = mh.ModelHolder(model=SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon),
                                gen_fit_data_func=mh.gen_SVR_fit_data,
                                gen_predict_data_func=mh.gen_SVR_predict_data)

    my_mh.gen_fit_data(ts_fit_start, ts_fit_end)
    my_mh.scale_fit_vars()
    my_mh.fit()

    my_mh.gen_predict_data(ts_predict_start, ts_predict_end)
    my_mh.predict()

    return my_mh.predict_y_dict


def get_tomorrow_timesteps(today):
    ts_today_h1 = dt.datetime(today.year, today.month, today.day, 1)
    ts_predict_start = ts_today_h1 + dt.timedelta(hours=24)
    ts_predict_end = ts_today_h1 + dt.timedelta(hours=2*24-1)

    return ens.gen_hourly_timesteps(ts_predict_start, ts_predict_end)


if __name__ == "__main__":
    main(sys.argv[1:])



