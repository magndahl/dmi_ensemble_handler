# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 12:18:38 2016

@author: Magnus Dahl
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import sql_tools as sq
import ensemble_tools as ens

specific_heat_water = 1.17e-6 # MWh/kgKelvin


class PumpToyModel(object):
    def __init__(self, delivered_heat, mass_flow_cap, T_sup_init=75, T_ret=40):
        self.T_ret = T_ret
        self.delivered_heat = delivered_heat
        self.mass_flow_cap = mass_flow_cap
        self.T_sup_init = T_sup_init
        self.mass_flow = np.zeros_like(mass_flow_cap)
        self.T_sup = T_sup_init*np.ones_like(delivered_heat)
        
    def calc_mass_flow_and_T_sup(self):
        mass_flow = self.delivered_heat/(specific_heat_water*(self.T_sup_init - self.T_ret))
        new_mass_flow = []
        new_T_sup = []
        for mf, mf_cap, Ts in zip(mass_flow, self.mass_flow_cap, self.T_sup):
            if mf > mf_cap:
                mf_reduction_factor = mf_cap/mf
                new_mass_flow.append(mf_cap)
                new_T_sup.append(Ts/mf_reduction_factor + self.T_ret*(mf_reduction_factor-1)/mf_reduction_factor)
            else:
                new_mass_flow.append(mf)
                new_T_sup.append(Ts)
        
      
        self.mass_flow = new_mass_flow
        self.T_sup = new_T_sup
 
               
mass_flow_cap_pct_of_full = 0.8
T_grnd = 6.4 # mean ground temperature over the period
ts1 = ens.gen_hourly_timesteps(dt.datetime(2015,12,17,1), dt.datetime(2016,1,15,0))
ts2 = ens.gen_hourly_timesteps(dt.datetime(2016,1,20,1), dt.datetime(2016,2,5,0))
all_ts = ts1 + ts2    
mean_price = np.concatenate([sq.fetch_price(ts1[0], ts1[-1]), sq.fetch_price(ts2[0], ts2[-1])]).mean()
          
def plot_const_vs_dym_cap(mass_flow_full_cap_cw):
    plt.close('all')   
    mass_flow_100pct_cap = mass_flow_full_cap_cw/specific_heat_water

    combined_conf_int = np.load('combined_conf_int.npz')['combined_conf_int']
    prod = np.concatenate([sq.fetch_production(ts1[0], ts1[-1]), sq.fetch_production(ts2[0], ts2[-1])])
    PTM_const = PumpToyModel(delivered_heat=prod, mass_flow_cap=mass_flow_100pct_cap*mass_flow_cap_pct_of_full*np.ones_like(prod))
    PTM_const.calc_mass_flow_and_T_sup()
    PTM_dyn = PumpToyModel(delivered_heat=prod, mass_flow_cap=mass_flow_100pct_cap*(1-(1-mass_flow_cap_pct_of_full)*combined_conf_int/combined_conf_int.max())*np.ones_like(prod))
    PTM_dyn.calc_mass_flow_and_T_sup()
    
    plt.figure(figsize=(20,10))
    plt.subplot(3,1,1)
    plt.plot_date(all_ts, PTM_const.mass_flow, 'r-', label='"Massflow" const cap')
    plt.plot_date(all_ts, PTM_dyn.mass_flow, 'g-', label='"Massflow" dyn cap')
    plt.plot_date(all_ts, PTM_const.mass_flow_cap, 'k--', label='Constant cap')
    plt.plot_date(all_ts, PTM_dyn.mass_flow_cap, 'y--', label='Dynamic cap')
    plt.ylabel("Massflow [kg/hour]")
    plt.legend(loc=4)
    
    plt.subplot(3,1,2)
    plt.plot_date(all_ts, PTM_const.T_sup, 'r-', label='T_sup constant cap')
    plt.plot_date(all_ts, PTM_dyn.T_sup, 'g-', label='T_sup dynamic cap')
    plt.ylabel('T_sup [degree C]')
    plt.legend()
    
    plt.subplot(3,1,3)
    reduced_heat_loss_pct = (np.array(PTM_dyn.T_sup) - T_grnd)/(np.array(PTM_const.T_sup) - T_grnd)
    plt.plot_date(all_ts, reduced_heat_loss_pct, 'r')
    hours_with_reduced_heat_loss = len(np.where(reduced_heat_loss_pct!=1)[0])
    average_heat_loss_reduction = reduced_heat_loss_pct[np.where(reduced_heat_loss_pct!=1)].mean()
    estimated_savings_MWh = 571e3*(1-average_heat_loss_reduction)*float(hours_with_reduced_heat_loss)/(365*24) # the 571e3 MWh corresponds toe 19% of the total annual producion
    estimated_savings_DKK = estimated_savings_MWh*mean_price
    
    plt.text(dt.datetime(2015,12,17,12), 0.98,\
             "Hours with reduced heat loss: %i\nEstimated saved: %2.1f  MWh\nEstimated saved: %2.2f DKK"\
             %(hours_with_reduced_heat_loss, estimated_savings_MWh, estimated_savings_DKK))
    
    plt.ylabel('Heatloss_const/heatloss_dyn')
    plt.savefig('figures/toymodel/const_vs_dyn_cap%i.pdf'%mass_flow_full_cap_cw)

        
                
        