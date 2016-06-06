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

    now = dt.datetime.now()

    now_whole_hour = dt.datetime(now.year, now.month, now.day, now.hour)
    ts_start = dt.datetime(2016,1,1,1)
    ts_end = now_whole_hour + dt.timedelta(hours=-1)
    try:
        prod = sq.fetch_production(ts_start, ts_end)
    except:
        print "Error in fetching production, ts_end=%s, nothing saved" % str(ts_end)
        return

    timesteps = pd.date_range(ts_start, ts_end, freq='H')

    timeseries = pd.Series(prod, index=timesteps)
    timeseries.to_pickle('/home/magnus/local_production/production_ts.pkl')
    print "Succesfully fetchted production from %s to %s" % (ts_start, ts_end)
    print "Production saved to '/home/magnus/local_production/production_ts.pkl"


if __name__ == "__main__":
    main(sys.argv[1:])
