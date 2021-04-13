# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:56:21 2018

@author: ACN980
"""

# Combining the results to only keep the most relevant

import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from Boostrap_JointProb import  bootstrap_stats_full

time_window = 30
#hazard = 'EQ'
result_filename = 'result.csv'

#Reading result csv files
fn_TC_res = os.path.join(r'H:\CONSECUTIVE\MODELING\OUTPUT\MONO', 'TC', str(time_window)+'_days', result_filename)
fn_EQ_res = os.path.join(r'H:\CONSECUTIVE\MODELING\OUTPUT\MONO', 'EQ', str(time_window)+'_days', result_filename)
fn_TC_EQ_res = os.path.join(r'H:\CONSECUTIVE\MODELING\OUTPUT\MULTI', 'EQ_TC', str(time_window)+'_days', result_filename)

# Set GID_2 as index
TC = pd.read_csv(fn_TC_res, index_col='index')
TC.drop(labels = 'nb_events', axis = 1, inplace = True)
TC.set_index('GID_2', drop = True, inplace = True)

EQ = pd.read_csv(fn_EQ_res, index_col='index')
EQ.drop(labels = 'nb_events', axis = 1, inplace = True)
EQ.set_index('GID_2', drop = True, inplace = True)

TC_EQ = pd.read_csv(fn_TC_EQ_res, index_col='index')

# Remove labels which have both
final = TC_EQ.copy()
final.drop(labels = 'nb_events', axis = 1, inplace = True)
final.set_index('GID_2', drop = True, inplace = True)

EQ.drop(final.index, inplace = True)
TC.drop(final.index, inplace = True)

#fn_out = r'F:\CONSECUTIVE\PAPER\CONF\Paper_Ruiter_Global'
#EQ.to_csv(os.path.join(fn_out,str(time_window)+'d','eq.csv'))
#TC.to_csv(os.path.join(fn_out,str(time_window)+'d','tc.csv'))
#final.to_csv(os.path.join(fn_out,str(time_window)+'d','final.csv'))

#%% Do the histogram

#We combine the dataframes
histo_30 = pd.concat([EQ,TC,final], axis=0)
histo_3 = pd.concat([EQ,TC,final], axis=0)

test_dt1 = histo_3.copy()
test_dt2 = histo_30.copy()

bins = np.linspace(0, 1, 100)
max_val = max([test_dt2['cond_prob'].max(), test_dt1['cond_prob'].max()])
max_val = np.round(max_val, decimals=1)

fig, ax1 = plt.subplots()
#Making a histogram
#plt.figure()
ax1.hist(test_dt2['cond_prob'], bins, facecolor='g', alpha=0.5) 
ax1.hist(test_dt1['cond_prob'], bins, facecolor='b', alpha=0.5)
ax1.axis([0, max_val, 0, 1600])
#plt.xlabel('Conditional Probability: P(EQ$_{t-n}$|EQ)')
plt.xlabel('Conditional Probability: P(H$_{t-n}$|H$_{t}$)')
plt.ylabel('Frequency')

## These are in unitless percentages of the figure size. (0,0 is bottom left)
#left, bottom, width, height = [0.5, 0.4, 0.3, 0.3]
#ax2 = fig.add_axes([left, bottom, width, height])
#
#ax2.hist(test_dt2['cond_prob'], bins, facecolor='g', alpha=0.5) 
#ax2.hist(test_dt1['cond_prob'], bins, facecolor='b', alpha=0.5)
#ax2.axis([0.01, 0.55, 0, 100])

plt.show()

##fig.savefig(r'F:\CONSECUTIVE\MODELING\OUTPUT\MONO\EQ\3_30days_intersectEQ.png', bbox_inches = 'tight', dpi = 300)
#fig.savefig(r'F:\CONSECUTIVE\PAPER\CONF\Paper_Histo_Figure.png', bbox_inches = 'tight', dpi = 400)

#%% Do the uncertainty around it -  for this, we modify the figure style a bit

# fig, ax1 = plt.subplots()
# #Making a histogram
# counts, bins, bars = plt.hist(test_dt1['cond_prob'], bins = bins, density= True, stacked = True)
# plt.show()

# counts2, bins2 = np.histogram(test_dt1['cond_prob'], bins = bins, density= True)
# #Bootstrap
# n_boot = 5000
# full_boot = np.random.choice(test_dt1['cond_prob'], size=(1,test_dt1['cond_prob'].shape[0]), replace=True, p=None)
# hist, bin_edges = np.histogram(full_boot, bins=bins, density=True)
# all_boot = pd.DataFrame(data = hist)
# for i in np.arange(2,n_boot+1):
#     full_boot = np.random.choice(test_dt1['cond_prob'], size=(1,test_dt1['cond_prob'].shape[0]), replace=True, p=None)
#     hist, bin_edges = np.histogram(full_boot, bins=bins, density=True)
#     all_boot = pd.concat([all_boot, pd.DataFrame(data = hist)], axis = 1)

# data_stats = pd.DataFrame(data=None, columns = ['alpha5', 'alpha95'])
# for i in all_boot.index.values:
#     data_stats.loc[i,'alpha5'], data_stats.loc[i,'alpha95'] = bootstrap_stats_full(pd.Series(all_boot.loc[i,:].values), 5)

# style = {}
# style['marker'] = 'o'
# style['linestyle'] = '-'
# style['markersize'] = 2
# style['linewidth'] = 1

# font = {'family': 'sans-serif',
#         'weight': 'normal',
#         'size': 7,
#         }

# plt.rc('font', **font)


# fig, ax = plt.subplots(1, 1, figsize=[4, 4])
# ax.plot(bins[:-1], counts, marker = style['marker'], linestyle = style['linestyle'], markersize = style['markersize'], linewidth = style['linewidth'], color = 'k', label = 'from simulated AM')
# ax.plot(bins[:-1], data_stats.loc[:,'alpha5'], '--k', label = '95% CI')
# plt.legend(loc=(0.12,0.77))
# ax.plot(bins[:-1], data_stats.loc[:,'alpha95'], '--k')




#     plt.figure()
#     counts, bins, bars = plt.hist(Joint_AM['joint'], bins = range(37), density= True)
#     plt.plot(bins[:-1],prob_90.loc[:,'3'], '-ok')
#     plt.show()
    
#     counts2, bins2 = np.histogram(Joint_AM['joint'], bins = bins, density= True)
    
#     #Bootstrap
#     n_boot = 5000
#     full_boot = np.random.choice(Joint_AM['joint'], size=(1,Joint_AM['joint'].shape[0]), replace=True, p=None)
#     hist, bin_edges = np.histogram(full_boot, bins=bins, density=True)
#     all_boot = pd.DataFrame(data = hist)
    
#     for i in np.arange(2,n_boot+1):
#         full_boot = np.random.choice(Joint_AM['joint'], size=(1,Joint_AM['joint'].shape[0]), replace=True, p=None)
#         hist, bin_edges = np.histogram(full_boot, bins=bins, density=True)
#         all_boot = pd.concat([all_boot, pd.DataFrame(data = hist)], axis = 1)
        
#     data_stats = pd.DataFrame(data=None, columns = ['alpha5', 'alpha95'])
#     for i in all_boot.index.values:
#         data_stats.loc[i,'alpha5'], data_stats.loc[i,'alpha95'] = bootstrap_stats_full(pd.Series(all_boot.loc[i,:].values), 5)

























 
