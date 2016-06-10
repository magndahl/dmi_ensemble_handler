#!/home/magnus/anaconda2/bin/python

import sys, getopt
import sql_tools as sq
import ensemble_tools as ens
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # this is needed to rund under cron
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import plotly.plotly as ply
import plotly.tools as tls
import plotly.graph_objs as go
from sklearn.svm import SVR
import ensemble_tools as ens
from error_tools import rmse, mae, mape
import ModelHolder as mh
from SVR_forecast_to_plotly import forecast_tomorrow, get_tomorrow_timesteps

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

    plyfigEO3 = plot_EO3_today_forecast(today)
    ply.plot(plyfigEO3, filename="Today's EO3 forecast", auto_open=False)

    plyfig = plot_today_forecast(today)
    ply.plot(plyfig, filename="Today's SVR forecast", auto_open=False)


def plot_today_forecast(today):
    fig = plt.figure()
    yesterday = mh.h_hoursbefore(today, 24)
    fc_dict = forecast_tomorrow(yesterday)
    mean_forecast = fc_dict['ens_mean']
    ens_forecast = np.array([fc_dict['ens_%i'%i] for i in range(25)]).transpose()
    ts = get_tomorrow_timesteps(yesterday)

    full_prod = pd.read_pickle('/home/magnus/local_production/production_ts.pkl')
    last_avail_prod_timestep = full_prod.index[-1]
    ts_last_prod = min(ts[-1], last_avail_prod_timestep)
    prod = sq.load_local_production(ts[0], ts_last_prod)
    plt.plot_date(ens.gen_hourly_timesteps(ts[0], ts_last_prod), prod, 'k-', lw=2, label='Realized production')
    plt.plot_date(ts, ens_forecast, '-', color='0.25', lw=0.3)
    plt.plot_date(ts, mean_forecast, 'b-', lw=2, label='Forecast')
    plt.ylabel("Production [MW]")
    plt.title("Today: " + str(today))

    err = mean_forecast[:len(prod)] - prod
    err_string = """RMSE = %2.2f MW\nMAE = %2.2f MW\nMAPE = %2.2f %%""" \
                % (rmse(err), mae(err), 100*mape(err, prod))
    plt.annotate(err_string, xy=(0.78, 0.80), xycoords='axes fraction')

    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig['layout']['showlegend'] = True
    plotly_fig['layout']['autosize'] = True
    plotly_fig['layout']['legend'] = dict(x=0.75, y=0.05)
    for d in plotly_fig.data:
        if d['name'][0:5]=='_line':
            d['showlegend'] = False

    return plotly_fig


def plot_EO3_today_forecast(today):

    fig = plt.figure()
    yesterday = mh.h_hoursbefore(today, 24)
    ts = get_tomorrow_timesteps(yesterday)
    predicted = sq.fetch_EO3_9oclock_forecast(ts[0], ts[-1])

    full_prod = pd.read_pickle('/home/magnus/local_production/production_ts.pkl')
    last_avail_prod_timestep = full_prod.index[-1]
    ts_last_prod = min(ts[-1], last_avail_prod_timestep)
    prod = sq.load_local_production(ts[0], ts_last_prod)
    plt.plot_date(ens.gen_hourly_timesteps(ts[0], ts_last_prod), prod, 'k-', lw=2, label='Realized production')
    plt.plot_date(ts, predicted, 'r-', lw=2, label='Forecast')

    plt.ylabel("Production [MW]")
    plt.title("Today: " + str(today))

    err = predicted[:len(prod)] - prod
    err_string = """RMSE = %2.2f MW\nMAE = %2.2f MW\nMAPE = %2.2f %%""" \
                % (rmse(err), mae(err), 100*mape(err, prod))
    plt.annotate(err_string, xy=(0.78, 0.80), xycoords='axes fraction')
    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig['layout']['showlegend'] = True
    plotly_fig['layout']['autosize'] = True
    plotly_fig['layout']['legend'] = dict(x=0.75, y='0.05')

    return plotly_fig



if __name__ == "__main__":
    main(sys.argv[1:])


