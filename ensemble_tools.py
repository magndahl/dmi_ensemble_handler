# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:15:04 2015

@author: Magnus Dahl
"""

import numpy as np
import datetime as dt

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
    
    
def timestamp_str(timestamp):
    """ Takes a datetime object at returns a string on the form:
        yyyymmddhh. Auxilary funtion to get_most_recent_filename(...).
        
        """
        
    return '%02i%02i%02i%02i' % (timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
    

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