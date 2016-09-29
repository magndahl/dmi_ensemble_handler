#!/home/magnus/anaconda2/bin/python

import sys, getopt
import sql_tools as sq
import ensemble_tools as ens
import datetime as dt
import matplotlib
matplotlib.use("Agg") # this is needed to rund under cron
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import plotly.plotly as ply
import plotly.tools as tls
import plotly.graph_objs as go
from error_tools import rmse, mae, mape


def main(argv):
    if len(argv)==0:
        print "no argument given, defaults to online mode"
        today = dt.datetime.now().date()
    elif argv[0]=='online':
        today = dt.datetime.now().date()
        print today
    else:
        try:
            today = dt.datetime.strptime(argv[0], '%Y-%m-%d').date()
            print today
        except ValueError:
            print "Argument must be 'online' or a valid date in yyyy-mm-dd format."
            sys.exit()

    # plot_historic_EO3_forecast and benchmark
    fig = plt.figure()
    ts_end = dt.datetime(today.year, today.month, today.day, 0)
    ts = ens.gen_hourly_timesteps(dt.datetime(2016,2,26,1), ts_end)

    predicted = sq.load_local_EO3_9fc(ts[0], ts[-1])
    prod = sq.load_local_production(ts[0], ts[-1])
    plt.plot_date(ts, prod, 'k-', lw=2, label='Realized production')
    plt.plot_date(ts, predicted, 'r-', lw=2, label='Forecast')
    plt.ylabel("Production [MW]")
    err = predicted - prod
    err_string = """RMSE = %2.2f MW\nMAE = %2.2f MW\nMAPE = %2.2f %%""" \
                % (rmse(err), mae(err), 100*mape(err, prod))
    plt.annotate(err_string, xy=(0.80, 0.80), xycoords='axes fraction')

    plotly_fig_hist = tls.mpl_to_plotly(fig)
    plotly_fig_hist['layout']['showlegend'] = True
    plotly_fig_hist['layout']['autosize'] = True
    plotly_fig_hist['layout']['legend'] = dict(x=0.75, y='0.05')

    ply.plot(plotly_fig_hist, filename="EO3 forecast history", auto_open=False)


    # Plot prediction for yesterday + benchmark
    ts_start_yest = dt.datetime(today.year, today.month, today.day, hour=1)\
            +dt.timedelta(hours=-24)
    ts_end_yest = ts_start_yest + dt.timedelta(hours=23)

    predicted_yest = sq.fetch_EO3_9oclock_forecast(ts_start_yest, ts_end_yest)
    actual_prod_yest = sq.fetch_production(ts_start_yest, ts_end_yest)

    fig_yest = plt.figure()
    plt.plot_date(ens.gen_hourly_timesteps(ts_start_yest, ts_end_yest),\
            predicted_yest, 'r-', lw=2, label='Forecast')
    plt.plot_date(ens.gen_hourly_timesteps(ts_start_yest, ts_end_yest),\
            actual_prod_yest, 'k-', lw=2, label='Realized production')
    plt.ylabel("Production [MW]")
    plt.title("Yesterday: " + str(today+dt.timedelta(days=-1)))
    err = predicted_yest-actual_prod_yest
    err_string = """RMSE = %2.2f MW\nMAE = %2.2f MW\nMAPE = %2.2f %%""" \
                % (rmse(err), mae(err), 100*mape(err, actual_prod_yest))
    plt.annotate(err_string, xy=(0.80, 0.80), xycoords='axes fraction')
    plotly_fig_yest = tls.mpl_to_plotly(fig_yest)
    plotly_fig_yest['layout']['showlegend'] = True
    plotly_fig_yest['layout']['autosize'] = True
    plotly_fig_yest['layout']['legend'] = dict(x=0.75, y='0.05')
    url_yest = ply.plot(plotly_fig_yest, filename="Yesterday's EO3 forecast", auto_open=False)



    # Plot prediction for tomorrow
    ts_start = dt.datetime(today.year, today.month, today.day, hour=1)\
            +dt.timedelta(hours=24)
    ts_end = ts_start + dt.timedelta(hours=23)

    predicted = sq.fetch_EO3_9oclock_forecast(ts_start, ts_end)

    fig = plt.figure()
    plt.plot_date(ens.gen_hourly_timesteps(ts_start, ts_end), predicted,\
            'r-', lw=2, label='Forecast')
    plt.ylabel("Production [MW]")
    plt.title("Tomorrow: " + str(today+dt.timedelta(days=1)))
    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig['layout']['showlegend'] = True
    plotly_fig['layout']['autosize'] = True
    plotly_fig['layout']['legend'] = dict(x=0.75, y=0.05)
    url = ply.plot(plotly_fig, filename='EO3 forecast for tomorrow', auto_open=False)



if __name__ == "__main__":
    main(sys.argv[1:])
