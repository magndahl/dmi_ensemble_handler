# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:30:28 2016

@author: Magnus Dahl
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import datetime as dt
import gurobipy as gb
import ensemble_tools as ens
import sql_tools as sq

plt.close('all')
ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
all_ts = ts1 + ts2

specific_heat_water = 1.17e-6 # MWh/kgKelvin
density_water = 980 # kg/m3 at 65 deg C
T_ret = 36.5

PI_T_sup = '4.146.120.29.HA.101'
PI_Q = 'K.146A.181.02.HA.101'

df = pd.DataFrame()
df['T_sup']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts1[0], \
            ts1[-1]),sq.fetch_hourly_vals_from_PIno(PI_T_sup, ts2[0], ts2[-1])])
df['Q']=np.concatenate([sq.fetch_hourly_vals_from_PIno(PI_Q, ts1[0], ts1[-1]),\
            sq.fetch_hourly_vals_from_PIno(PI_Q, ts2[0], ts2[-1])])
df['ts'] = all_ts
df['cons'] = specific_heat_water*density_water*df['Q']*(df['T_sup']-T_ret)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
cond_df = df#[df['Q'] > 288]
ax1.plot_date(np.array(cond_df['ts']), np.array(cond_df['Q']), 'b')

ax2.plot_date(np.array(cond_df['ts']), np.array(cond_df['T_sup']), 'r-')

plt.figure()
plt.plot_date(df['ts'], df['cons'], 'g-')

plt.figure()
plt.scatter(df['T_sup'], df['Q'], c=df['cons'])
plt.colorbar()

#plt.figure()
#plt.scatter(df['T_sup'], df['cons'])
#
#plt.figure()
#plt.scatter(df['Q'], df['cons'])

#%% area1
low_flow = df['Q']<300
low_T = df['T_sup']<70
df1 = df[low_flow & low_T]

#%% area2
medium_flow = (df['Q']>295) & (df['Q'] <340)
medium_T = df['T_sup'] < 73
df2 = df[medium_flow & medium_T]

#area3
high_flow = (df['Q']>340) & (df['Q'] <360)
df3 = df[high_flow]

res=sm.OLS(df3['Q'], sm.add_constant(df3['T_sup'])).fit()

# simple model
T1 = 68.5
a2 = 15.5
a3 = 2.1
b2 = 295-a2*T1
b3 = 340-a3*71.4
plt.plot(T1*np.ones(len(df1)), df1['Q'], 'k-', lw=3)
plt.plot(df2['T_sup'], a2*df2['T_sup']+b2, 'k-', lw=3)
plt.plot(df3['T_sup'], a3*df3['T_sup']+b3, 'k-', lw=3)
plt.plot(df[df['T_sup']>78]['T_sup'], 360*np.ones_like(df[df['T_sup']>78]['T_sup']),'k-', lw=3)

#%% This is where i calculate the test the idealized version of the model unsuccesful gurobi attempt
#Q_ub = 360
#m = gb.Model()
#Q = m.addVar(lb=0, ub=Q_ub, name='Q')
#T_sup = m.addVar(lb=T1, ub=125, name='T_sup')
#m.update()
#m.setObjective(m.getVarByName('T_sup'))
#m.modelSense = gb.GRB.MINIMIZE
#lhs = gb.QuadExpr()
#lhs.addTerms(specific_heat_water*density_water, m.getVarByName('Q'),m.getVarByName('T_sup'))
#lhs.addTerms(-specific_heat_water*density_water*T_ret, m.getVarByName('Q'))
#m.addQConstr(lhs=lhs, sense=gb.GRB.EQUAL, rhs=10, name='energy_bal')
#m.addConstr(m.getVarByName('Q')<=15.5*m.getVarByName('T_sup') - 766.75)
#m.addConstr(m.getVarByName('Q')<=2.1*m.getVarByName('T_sup') + 190)
#
#m.update()

def Q_from_cons_lin_piece(cons, a, b):
    B = -(b+a*T_ret)/a
    C = -cons/(specific_heat_water*density_water)
    A = 1/a

    Qplus = (-B+np.sqrt(B**2 - 4*A*C))/(2*A)
    
    return Qplus
    

def get_Tsup_and_Q(cons, Q_ub):
    # try lowes possible T    
    Q = cons/(specific_heat_water*density_water*(T1 - T_ret))
    if Q <= 295:
        return T1, Q
    elif Q > 295:
        Q = Q_from_cons_lin_piece(cons, a2, b2)
        if Q <= Q_ub*(340./360):
            T = (Q - b2)/a2  
            return T, Q
        elif Q >= Q_ub*(340./360):
            b3_adjusted = b3 + (Q_ub*(340./360) - 340)
            Q = Q_from_cons_lin_piece(cons, a3, b3_adjusted)
            if Q <= Q_ub:
                T = (Q - b3_adjusted)/a3
                return T, Q
            elif Q > Q_ub:
                Q = Q_ub
                T = cons/(specific_heat_water*density_water*Q) + T_ret
                return T, Q

const_Q_ub = 360
Q_const_cap = []
T_sup_const_cap = []
for c in df['cons']:
    T, Q = get_Tsup_and_Q(c, const_Q_ub)
    Q_const_cap.append(Q)
    T_sup_const_cap.append(T)
    
model_conf_int = np.load('combined_conf_int.npz')['combined_conf_int']
assert(list(np.load('combined_conf_int.npz')['timesteps'])==all_ts), "confidence intervals don't have matching time steps"

Q_dyn_cap = []
T_sup_dyn_cap = []
dyn_Q_ub = []
for c, model_uncertainty in zip(df['cons'], model_conf_int):
    Q_ub = 410 - (410-const_Q_ub)*(model_uncertainty/np.max(model_conf_int))
    dyn_Q_ub.append(Q_ub)
    T, Q = get_Tsup_and_Q(c, Q_ub)
    Q_dyn_cap.append(Q)
    T_sup_dyn_cap.append(T)
    
plt.figure()
plt.subplot(3,1,1)
plt.plot_date(all_ts, Q_const_cap, 'r-')
plt.plot_date(all_ts, Q_dyn_cap, 'g-')
plt.plot_date(all_ts, const_Q_ub*np.ones(len(all_ts)), 'k--')
plt.plot_date(all_ts, dyn_Q_ub, 'y--')

plt.subplot(3,1,2)
plt.plot_date(all_ts, T_sup_const_cap, 'r-')
plt.plot_date(all_ts, T_sup_dyn_cap, 'g-')

plt.subplot(3,1,3)
T_grnd = 6.4
heat_loss_reduction = 1 - (np.array(T_sup_dyn_cap) - T_grnd)/(np.array(T_sup_const_cap) - T_grnd)
plt.plot(all_ts, heat_loss_reduction)