#!/home/magnus/anaconda2/bin/python

import sql_tools as sq
import sys
import datetime as dt
import pandas as pd

def main(argv):
    """ This script fetches the total production in
        the Aarhus DH system from 2016/1/1 h=1 to one
        hour ago (from now) with hourly values.
        The result is saved to
        /home/magnus/local_production/production_ts.pkl

        The script take no arguments and is meant to be
        run at 30 minutes past each hour through crontab.

        """

    already_saved = pd.read_pickle('/home/magnus/local_EO3_9fc/forecast_ts.pkl')

    now = dt.datetime.now()

    today_at_1oclock = dt.datetime(now.year, now.month, now.day, 1)
    ts_start = already_saved.index[-1] + dt.timedelta(hours=1)
    ts_end = ts_start + dt.timedelta(hours=23)
    try:
        pred = sq.fetch_EO3_9oclock_forecast(ts_start, ts_end)
    except:
        print "Error in fetching forecast, ts_end=%s, nothing saved" % str(ts_end)
        return

    timesteps = pd.date_range(ts_start, ts_end, freq='H')
    timeseries = pd.Series(pred, index=timesteps)

    new_ts = pd.concat([already_saved, timeseries])
    new_ts.to_pickle('/home/magnus/local_EO3_9fc/forecast_ts.pkl')


if __name__ == "__main__":
    main(sys.argv[1:])
