# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:29:13 2015

@author: Magnus
"""

import pymssql
import numpy as np

login_info = np.load('settings/forretningslag_login.npz')
srv = str(login_info['server'])
usr = str(login_info['user'])
psw = str(login_info['password'])

def connect():
    conn = pymssql.connect(server=srv, user=usr, password=psw)
    
    return conn
    

def extractdata(conn, sql_query):
    curs = conn.cursor()
    curs.execute(sql_query)
    
    return curs.fetchall()

