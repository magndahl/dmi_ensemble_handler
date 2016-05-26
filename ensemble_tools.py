# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:15:04 2015

@author: Magnus Dahl
"""

import numpy as np
import datetime as dt
import pandas as pd
import dateutil.rrule
import sql_tools as sq


# properties of the DMI data
DMI_update_interval = 6 # every 6 hours at 00, 06, 12 and 18
number_of_timesteps = 55 # in every forecast
number_of_ensembles = 25
datapath = 'Q:/Weatherdata/DMI_ensembler/'


field_map = {'Tout':1, 'vWind':2, 'hum':3, 'sunRad':4, 'Tgrnd0cm':5}

def get_most_recent_filename(timestamp, data_delay=5):
    """ This function takes a timestamp for the endpoint of the 
        hour of interest along with a number of hours the dmi ensembles
        are delayed from the timestamp on the filename. It returns the
        name of the file with the most recent forecast for the given 
        timestep.
        
        """
        
    delay_hours_ago = timestamp + dt.timedelta(hours=-data_delay)
    data_file_timestamp = dt.datetime(delay_hours_ago.year,
                                      delay_hours_ago.month,
                                      delay_hours_ago.day,
                                      DMI_update_interval*\
                                      np.floor_divide(delay_hours_ago.hour,\
                                                      DMI_update_interval))
                                                      
    return 'Aarhus_' + timestamp_str(data_file_timestamp)
    

def get_avail_at10_thedaybefore_filename(timestamp):
    """ This function takes a timestamp for the endpoint of the 
        hour of interest. It returns the filename for the forecast file
        that was available at 10 am the previous day. This is
        the 00 hour forecast with the current delay.
        
        """
    
    timestamp_startof_hour = timestamp + dt.timedelta(hours = -1)        
    timestamp_day_before = timestamp_startof_hour + dt.timedelta(days = -1)
    data_file_timestamp = dt.datetime(timestamp_day_before.year,
                                      timestamp_day_before.month,
                                      timestamp_day_before.day, 0)
                                      
    return 'Aarhus_' + timestamp_str(data_file_timestamp)
    

def gen_timeseries(field, timesteps, pointcode=71699, \
                   get_filename_func=get_most_recent_filename, \
                   get_filenam_func_kwargs=None):
    time_series = np.zeros((len(timesteps),25))
    for ts, i in zip(timesteps, range(len(timesteps))):
        if in_dailightsavings_period(ts):
            ts = ts + dt.timedelta(hours=-1) # this compensates for the fact that DMI always uses UTC+1 timestamps
        filename = get_filename_func(ts)      
        searchstring = gen_searchstring_pointcode(field, pointcode)
        try:
            line_number = find_linenumber(searchstring, filename)
            data_arr = load_data(line_number, filename)
        except:
            print "Warning: file %s not found, tries the previous forecast instead!"%filename
            filename = get_previous_DMI_filename(filename)
            line_number = find_linenumber(searchstring, filename)
            data_arr = load_data(line_number, filename)
        time_series[i] = data_arr[row_number_by_timestamp(ts,data_arr), 1:]
        
    return time_series
        
        
def save_most_recent_timeseries(fieldname, ts_start, ts_end, pointcode=71699, \
                                savepath='time_series/'):
    filename = ''.join([fieldname, '_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end)])
    
    timesteps = gen_hourly_timesteps(ts_start, ts_end)
    ens_timeseries = gen_timeseries(field_map[fieldname], timesteps, pointcode)                
    np.save(savepath+filename, ens_timeseries)
    print "Saved file: %s"%filename
    

def save_avail_at10_timeseries(fieldname, ts_start, ts_end, pointcode=71699):
    savepath='time_series/avail_at10_thedaybefore/'
    filename = ''.join([fieldname, '_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end)])
    
    timesteps = gen_hourly_timesteps(ts_start, ts_end)
    ens_timeseries = gen_timeseries(field_map[fieldname], timesteps, pointcode,
                                    get_filename_func=get_avail_at10_thedaybefore_filename)                
    np.save(savepath+filename, ens_timeseries)
    print "Saved file: %s"%filename
    

def save_ens_mean_avail_at10_series(ts_start, ts_end, pointcode=71699):
    savepath='time_series/avail_at10_thedaybefore/ens_means/'
    suffix = ''.join(['_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end), '.npy'])
    for v in ['Tout', 'hum', 'vWind', 'sunRad']:
        try:
            ens_data = np.load('time_series/avail_at10_thedaybefore/' + v + suffix)
        except:
            print "Ensemble times series not found. Generating more: "
            save_avail_at10_timeseries(v, ts_start, ts_end, pointcode)
            ens_data = np.load('time_series/avail_at10_thedaybefore/' + v + suffix)
        
        if v=='Tout':
            ens_data = Kelvin_to_Celcius(ens_data) ## convert the temperature to celcius
        
        hourly_mean = ens_data.mean(axis=1)
        np.save(savepath + v + suffix, hourly_mean)
        print "Saved files: " + savepath + v + suffix
        

def load_ens_avail_at10_series(ts_start, ts_end, varname, pointcode=71699):
    suffix = ''.join(['_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end), '.npy'])
    try:
        ens_data = np.load('time_series/avail_at10_thedaybefore/' + varname + suffix)
    except:
        print "Ensemble times series not found. Generating more: "
        save_avail_at10_timeseries(varname, ts_start, ts_end, pointcode)
        ens_data = np.load('time_series/avail_at10_thedaybefore/' + varname + suffix)
    
    if varname=='Tout':
        ens_data = Kelvin_to_Celcius(ens_data) ## convert the temperature to celcius
        
    return ens_data


def load_ens_mean_avail_at10_series(var, ts_start, ts_end, pointcode=71699):
    load_path = 'time_series/avail_at10_thedaybefore/ens_means/'
    suffix = ''.join(['_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end), '.npy'])                                                            
    try:
        ens_mean = np.load(load_path + var + suffix)
    except:
        print "Ensemble mean time series not found. Generating more: "
        save_ens_mean_avail_at10_series(ts_start, ts_end, pointcode=pointcode)
        ens_mean = np.load(load_path + var + suffix)
    
    return ens_mean
    

def save_ens_mean_series(ts_start=dt.datetime(2015,12,16,1),\
                         ts_end=dt.datetime(2016,1,15,0), pointcode=71699, 
                         savepath='time_series/ens_means/'):
    """ Note, that the initial time stamp of the save time series is 24 hours
        later that ts_start because some of the time series have been averaged
        over the previous 24 hours.
        
        """
        
    
    load_suffix = ''.join(['_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end), '.npy'])
    save_suffix = ''.join(['_geo', str(pointcode), '_',\
                            timestamp_str(ts_start+dt.timedelta(hours=24)), \
                           '_to_', timestamp_str(ts_end)])
                           
    for v in ['Tout', 'hum', 'vWind', 'sunRad']:
        try:
            ens_data = np.load('time_series/' + v + load_suffix)
        except:
            print "Ensemble times series not found. Generating more: "
            save_most_recent_timeseries(v, ts_start, ts_end, pointcode=pointcode)
            ens_data = np.load('time_series/' + v + load_suffix)
        
        if v=='Tout':
            ens_data = Kelvin_to_Celcius(ens_data) ## convert the temperature to celcius
            
        hourly_mean = ens_data.mean(axis=1)
        
        mean_last_24h = np.array([np.mean(hourly_mean[i:i+24]) for i in range(hourly_mean.shape[0]-24)])
        hourly_mean_excepday1 = hourly_mean[24:]
        
        np.save(savepath + v + save_suffix, hourly_mean_excepday1)
        np.save(savepath + v + 'avg24' + save_suffix, mean_last_24h)
        print "Saved files: " + savepath + v + save_suffix + "\n" + savepath + v + 'avg24' + save_suffix


def load_ens_timeseries_as_df(ts_start=dt.datetime(2015,12,17,1),\
                         ts_end=dt.datetime(2016,1,15,0), pointcode=71699, weathervars=['Tout', 'vWind']):
    load_suffix = ''.join(['_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end), '.npy'])
    df = pd.DataFrame()                     
    for v in weathervars:
        try:
            ens_data = np.load('time_series/' + v + load_suffix)
        except:
            print "Ensemble times series not found. Generating more: "
            save_most_recent_timeseries(v, ts_start, ts_end, pointcode=pointcode)
            ens_data = np.load('time_series/' + v + load_suffix)
        
        if v=='Tout':
            ens_data = Kelvin_to_Celcius(ens_data) ## convert the temperature to celcius
        
        for i in range(ens_data.shape[1]):
            key = v + str(i)
            df[key] = ens_data[:,i]
    return df


def repack_ens_mean_as_df(ts_start=dt.datetime(2015,12,17,1), ts_end=dt.datetime(2016,1,15,0),\
                          load_path='time_series/ens_means/', pointcode=71699):
    load_suffix = ''.join(['_geo', str(pointcode), '_', timestamp_str(ts_start), \
                        '_to_', timestamp_str(ts_end), '.npy'])
    weathervars = ['Tout', 'hum', 'vWind', 'sunRad']
    allvars = weathervars + [v + 'avg24' for v in weathervars]
    data_dict = {v:np.load(load_path + v + load_suffix) for v in allvars}
    data_dict['prod'] = sq.fetch_production(ts_start, ts_end)
    data_dict['(Tout-17)*vWind'] = (data_dict['Tout']-17)*data_dict['vWind']
    data_dict['(Toutavg-17)*vWindavg24'] = (data_dict['Toutavg24']-17)*data_dict['vWindavg24']   
    dataframe = pd.DataFrame(data_dict)  
    return dataframe     


def gen_hourly_timesteps(start, stop):
    return list(dateutil.rrule.rrule(dateutil.rrule.HOURLY, dtstart=start, until=stop))
    

def row_number_by_timestamp(timestamp, array):
    """ Takes array like the one returned from load_data, along
        with at timestep. Returns the row of said timestep.
        Creates runtime error if the timestep is not in the array.
        
        """
    return np.where(array[:,0]==int(timestamp_str(timestamp)))[0][0]
    
    
def timestamp_str(timestamp):
    """ Takes a datetime object at returns a string on the form:
        yyyymmddhh. Auxilary funtion to get_most_recent_filename(...).
        
        """
        
    return '%02i%02i%02i%02i' % (timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
    

def timestamp_from_str(ts_string):
    return dt.datetime.strptime(ts_string, "%Y%m%d%H")
    

def find_linenumber(searchstring, filename, path=datapath):
    """ Takes a searchstring and a filename and returns the number of the 
        first line in which the searchstring appears. Returns None if no
        match is found. Line numbering begins from 1, like in Atom.
        
        """
        
    with open(path+filename, 'r') as inF:
        for i, line in enumerate(inF, 1):
		if searchstring in line:
			return i

def load_data(title_linenumber, filename, path=datapath):
    """ Take a linenumber for the titleline and the filename. Returns
        the 25 ensembles as columns in a numpy array, 55 timesteps long.
        
        """    
    skip_head = title_linenumber
    max_rows = number_of_timesteps
    data = np.genfromtxt(path+filename, skip_header=skip_head, \
    		max_rows=max_rows)
      
    return data


def gen_searchstring_long_lat(field, longitude=10.13, lattitude=56.17):
    """ Generate search string to find the header line of a geographical point
        and a certain weather field (see the field_map dictionary.)
        Default point is Bronzealdertoften 22, 8210.
        
        """
        
    return str(field) + "   " + str(longitude) + "   " + str(lattitude)
 
 
def gen_searchstring_pointcode(field, pointcode=71699):
    """ Generate search string to find the header line of a geographical point
        from the DMI point code and a certain weather field (see the field_map 
        dictionary.)
        Default point is Bronzealdertoften 22, 8210. Find the point codes in 
        the data files.
        
        """
        
    return str(pointcode) + "   " + str(field)
    
    
def get_previous_DMI_filename(filename):
    """ This function takes the filename of one of the DMI ensemble
        files and returns the one that was published 6 hours earlier.
        
        """
        
    ts_string = filename[-10:]
    timestamp = timestamp_from_str(ts_string)
    prev_timestep = timestamp + dt.timedelta(hours=-6)
      
    return filename[:-10] + timestamp_str(prev_timestep)


def Kelvin_to_Celcius(array):
    return array - 273.15

def frac_to_percent(array):
    return 100.*array    


def ensemble_std(array):
    return array.std(axis=1)
    
    
def ensemble_abs_spread(array):
    return array.max(axis=1) - array.min(axis=1)
    
    
def in_dailightsavings_period(timestep):
    """ Returns true if the time steps in the Danish daylight savings period,
        Between 2014 and 2020. """
        
    dls_start_stop_dict = {2014:(dt.datetime(2014,3,30,2), dt.datetime(2014,10,26,3)),
                           2015:(dt.datetime(2015,3,29,2), dt.datetime(2015,10,25,3)),
                           2016:(dt.datetime(2016,3,27,2), dt.datetime(2016,10,30,3)),
                           2017:(dt.datetime(2017,3,26,2), dt.datetime(2017,10,29,3)),
                           2018:(dt.datetime(2018,3,25,2), dt.datetime(2018,10,28,3)),
                           2019:(dt.datetime(2019,3,31,2), dt.datetime(2019,10,27,3)),
                           2020:(dt.datetime(2020,3,29,2), dt.datetime(2020,10,25,3))}
    for year in dls_start_stop_dict.keys():        
        if dls_start_stop_dict[year][0] < timestep <= dls_start_stop_dict[year][1]:
            return True
    
    return False
    
