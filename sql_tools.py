# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:29:13 2015

@author: Magnus
"""

import pymssql
import numpy as np
from ensemble_tools import gen_hourly_timesteps

login_info = np.load('settings/forretningslag_login.npz')
srv = str(login_info['server'])
usr = str(login_info['user'])
psw = str(login_info['password'])

BBSyd_pi_dict = {'Tout':"'2.123.121.60.HA.101'",
                 'vWind':"'2.123.140.01.HA.101'",
                 'sunRad':"'2.123.137.01.HA.101'",
                 'hum':"'2.123.148.60.HA.101'",
                 'Tgrnd100':"'2.123.124.10.HA.101'"}

def connect():
    conn = pymssql.connect(server=srv, user=usr, password=psw)
    
    return conn
    

def extractdata(conn, sql_query):
    curs = conn.cursor()
    curs.execute(sql_query)
    
    return curs.fetchall()


def fetch_BrabrandSydWeather(weathervar, from_time, to_time):
    """ This function takes a weather variable as a string (from BBSyd_pi_dict)
        as well as first and last step timestep (as datetime objects).
        It returns the hourly time series from the Brabrand Syd Weather station.
        Note that this data has not been validated!
        
        """
        
    conn = connect()
    PInr = BBSyd_pi_dict[weathervar]
    sql_query = """USE [DM_VLP]
                    SELECT 
                       [TimeStamp],
                       [Value],
                       [Beskrivelse]
                    FROM [dbo].[Meteorologi]
                        WHERE PInr=%s 
                            AND TimeStamp BETWEEN '%s' AND  '%s'
                        ORDER BY TimeStamp"""% (PInr, str(from_time), str(to_time))

       
    data = extractdata(conn, sql_query)
    timestamps, values, description = zip(*data)
    assert(list(timestamps)==gen_hourly_timesteps(from_time, to_time)), "Timesteps are not hour by hour"
    
    return np.array(values, dtype=float)
    