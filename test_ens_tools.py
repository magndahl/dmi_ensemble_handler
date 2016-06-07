import ensemble_tools as ens
import datetime as dt
import sklearn

ts = ens.gen_hourly_timesteps(dt.datetime(2016,1,25,1), dt.datetime(2016,4,1,0))

timeseries = ens.gen_timeseries(2, ts)

print timeseries
