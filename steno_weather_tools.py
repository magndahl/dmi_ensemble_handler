# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:09:04 2016

@author: Magnus Dahl
"""

from datetime import datetime, timedelta
from pytz import timezone
import pytz
utc = pytz.utc
import time
import calendar
import numpy as np
from ensemble_tools import timestamp_str

### NOTE: I have a strong suspicion that when DUKS says that the time stamps
### in *lg or *vantagelog files are local time (UTC + 1 in winter), it
### is actually local time for Finland or something (UTC + 2 in winter)
## This I base on the fact that sunrise seems to come an hour late.
## So far these functions assume that DUKS are correct, so the time series
## generated are shifted by one hour compared to data from Brabrand Syd.

datapath='Q:/Weatherdata/Steno_weatherstation/'

def save_hourly_timeseries(months_years = [(12,2015), (1,2016)], new_data_avail=True):
    """ This function saves the hourly time series of outdoor temperature, wind speed,
        humidity and solar radiation from the Steno weather station. Possibly 
        spanning several months. The timesteps are also saved.
        
        """
        
    dat_arrays_list = []     
    for mnyr in months_years:
        month = mnyr[0]
        year = mnyr[1]
        if new_data_avail:
            create_timestamp_weather(month, year)
        data = load_timestamp_weather(month, year)
        local_ts = get_local_timesteps(data)
        TvWhsR_data = create_Tout_vWind_hum_sunRad_array(data)
        dat_arrays_list.append(agg_data_hourly(TvWhsR_data, local_ts))
        
    timestep_array = np.concatenate([d[1] for d in dat_arrays_list])
    full_timeseries_array = np.concatenate([d[0] for d in dat_arrays_list])
    filename = 'Steno_hourly_' + timestamp_str(timestep_array[0]) + '_to_' + timestamp_str(timestep_array[-1])
    np.savez(datapath+filename, timesteps=timestep_array, Tout_vWind_hum_sunRad=full_timeseries_array)    
    
    return                   


def load_timestamp_weather(month, year, path=datapath):
    """ Loads output file from create_timestamp_weather(month, year, path=datapath)
        into a numpy array.
        
        """
        
    filename = str(month) + str(year) + '_timestamp_weather.txt'
    data=np.genfromtxt(path+filename, skip_header=2)
    return data
    
    
def get_local_timesteps(data):
    """ takes data array output of load_timestamps_weather
        IMPORTANT: Assumes no daylight savings time!!
        
        """
    timezoneoffset = 1 # Assuming we are in Aarhus in winter!!
    UTC_timestamps = [datetime.utcfromtimestamp(ts) for ts in data[:,0]]
    assert(UTC_timestamps[0].month>=10 or UTC_timestamps[0].month<=3), " you need to adjust for daylight savings time!!"
    assert(UTC_timestamps[-1].month>=10 or UTC_timestamps[-1].month<=3), " you need to adjust for daylight savings time!!"
    local_timestamps = [ts + timedelta(hours=timezoneoffset) for ts in UTC_timestamps]
    return local_timestamps


def create_Tout_vWind_hum_sunRad_array(data):
    """ takes data array output of load_timestamps_weather
        
        """
    out = np.array([data[:,3], data[:,6], data[:,4], data[:,9]], dtype=float)
    
    return out.transpose()
    

def agg_data_hourly(Tout_vWind_hum_sunRad_array, local_timestamps):
    """ This funcction takes the output of create_Tout_vWind_hum_sunRad_array(data),
        along with the local_timestamps from get_local_timesteps(data). 
        The function bunches the data by hour and assigns the mean of each bunch
        the the hourly ENDPOINT of the time interval.
        It's not elegant but it gets the job done.
        """
        
    hourly_timesteps = gen_hourly_timesteps(local_timestamps)
    
    data_bunched_by_hour = []
    for i in range(len(hourly_timesteps)):
        if i==0:
            start_hour = hourly_timesteps[0] + timedelta(hours=-1)
        else:
           start_hour = hourly_timesteps[i-1]
        stop_hour = hourly_timesteps[i]
        
        indices = [local_timestamps.index(ts) for ts in local_timestamps if start_hour < ts <= stop_hour]
        data_bunched_by_hour.append(Tout_vWind_hum_sunRad_array[indices,:])
    
    hourly_data = np.zeros((len(hourly_timesteps), Tout_vWind_hum_sunRad_array.shape[1]), dtype=float)
    for i in range(len(hourly_timesteps)):
        if not any(np.isnan(data_bunched_by_hour[i].mean(axis=0))):        
            hourly_data[i,:] = data_bunched_by_hour[i].mean(axis=0)
        else:
            try:
                hourly_data[i,:] = hourly_data[i-1,:] # if data is missing for an hour, set to point for the previous hour
            except:
                print "Missing data: Data for hourly timestep " + str(hourly_timesteps[i]) + " set to 0."
                                
    return hourly_data, hourly_timesteps
    
    
def gen_hourly_timesteps(local_timestamps):
    ts_start = local_timestamps[0]
    number_of_hours = int(np.ceil((local_timestamps[-1]-ts_start).total_seconds()/3600))
    ts_start_next_hour = ts_start + timedelta(hours=1)
    ts_start_hour = datetime(ts_start_next_hour.year, ts_start_next_hour.month, ts_start_next_hour.day, ts_start_next_hour.hour)
    hourly_timesteps = [ts_start_hour + timedelta(hours=x) for x in range(number_of_hours)]
    
    return hourly_timesteps    


def create_timestamp_weather(month, year, path=datapath):
    """ credit for this function to the writers of DUKS_Steno_reader.py
    this function is a slightly modified version of that script.
    http://users-phys.au.dk/vejrdata/
    
    """
    
    month_year_string=str(month)+str(year)
    #
    vantage_input=path + str(month_year_string)+'vantagelog.txt'
    vantage_data = open(str(vantage_input), 'r')
    lg_input=path + str(month_year_string)+'lg.txt'
    lg_data = open(str(lg_input), 'r')
    #
    output_filename=path + str(month_year_string)+'_timestamp_weather.txt'
    output_file=open(str(output_filename),'w')
    output_file.writelines('# Input files: '+str(vantage_input)+' '+str(lg_input)+'\n')
    output_file.writelines('# UTC_stamp'+'\t'+'UTC_year_month_day_time+tz'+'\t'+'temp_out'+'\t'+'humidity'+'\t'+'barometer'+'\t'+'windspeed'+'\t'+'winddir'+'\t'+'rainlastmin'+'\t'+'solar_rad'+'\t'+'UV'+'\n')
    #
    vantage_lines = vantage_data.readlines()
    vantage_data.close() 
    vantage_data = []
    vantage_data = [line.split() for line in vantage_lines] 
    print(vantage_data[1])
    print "Number of lines in vantage_data:",len(vantage_data)#Number of lines
    print "Number of columns in vantage_data:",len(vantage_data[0])#Number of columns
    print "Columns in vantage_data: ",vantage_data[0]
    print ''
    #
    lg_lines = lg_data.readlines()
    lg_data.close() 
    lg_data = []
    lg_data = [line.split() for line in lg_lines] 
    print(lg_data[1])
    print "Number of lines in lg_data:",len(lg_data)#Number of lines
    print "Number of columns in lg_data:",len(lg_data[0])#Number of columns
    print "Columns in lg_data: ",lg_data[1]
    print ''
    #
    
    
    #Check lengths of 3 vantage files
    if len(vantage_data)==len(lg_data):
        print "All 3 vantage files have same number of lines: ",len(lg_data)
        #elif (len(vantage_data)<>len(lg_data)):
        #    print "WARNING: The 3 vantage files do NOT have same number of lines"
    else:
        print "WARNING: The 3 vantage files do NOT have same number of lines" 
        print ''   
        end_line_number=min(len(vantage_data),len(lg_data))
    #check date and time in 3 vantage files
    #
    for line_number in range(1,len(vantage_data)):
        #
        vantage_local_dt=vantage_data[line_number][0:5]
        vantage_local_day=vantage_data[line_number][0]
        vantage_local_month=vantage_data[line_number][1]
        vantage_local_year=vantage_data[line_number][2]
        vantage_local_hour=vantage_data[line_number][3]
        vantage_local_minute=vantage_data[line_number][4]
        #
        lg_local_dt=lg_data[line_number][0:5]
        """ 
        lg_day=lg_data[line_number][0]
        lg_month=lg_data[line_number][1]
        lg_year=lg_data[line_number][2]
        lg_hour=lg_data[line_number][3]
        lg_minute=lg_data[line_number][4]
        """ 
        #
        """ 
        indoor_day=indoor_data[line_number][0]
        indoor_month=indoor_data[line_number][1]
        indoor_year=indoor_data[line_number][2]
        indoor_hour=indoor_data[line_number][3]
        indoor_minute=indoor_data[line_number][4]
        """ 
        #
        ok_local_dt=0
        if vantage_local_dt==lg_local_dt:
            ok_local_dt=ok_local_dt+1
            #break
        elif vantage_local_dt<>lg_local_dt:  
            print "ERROR: vantage_local_dt<>lg_local_dt, line_number= ",line_number
            break
        else:  
            print "ERROR: DAY/TIME mismatch, line_number=",line_number
            break
            #
        local_dt = str(vantage_local_day)+' '+str(vantage_local_month)+' '+str(vantage_local_year)+' '+str(vantage_local_hour)+' '+str(vantage_local_minute)
        UTC_dt = pytz.timezone('Europe/Copenhagen').localize(datetime.strptime(str(local_dt),'%d %m %Y %H %M')).astimezone(pytz.utc)
        #print vantage_local_dt
        #print UTC_dt
        UTC_stamp = calendar.timegm(UTC_dt.timetuple())
        #print UTC_stamp
        temp_out=lg_data[line_number][5]
        humidity=lg_data[line_number][6]
        barometer=lg_data[line_number][8]
        windspeed=lg_data[line_number][9]
        winddir=lg_data[line_number][11]
        rainlastmin=lg_data[line_number][12]
        #
        solar_rad=vantage_data[line_number][5]
        UV=vantage_data[line_number][6]
        #
        #
        output_file.writelines(str(UTC_stamp)+'\t'+str(UTC_dt)+'\t'+str(temp_out)+'\t'+str(humidity)+'\t'+str(barometer)+'\t'+str(windspeed)+'\t'+str(winddir)+'\t'+str(rainlastmin)+'\t'+str(solar_rad)+'\t'+str(UV)+'\n')
            
    output_file.close() 
