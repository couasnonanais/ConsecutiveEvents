# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:13:03 2018

@author: Anton
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from datetime import datetime  
from datetime import timedelta 
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

#Plot time series with the disasters - selecting location of Pinatubo
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) #see: https://matplotlib.org/examples/color/named_colors.html


def is_empty(any_structure):
    if any_structure:
        #print('Structure is not empty.')
        return False
    else:
        #print('Structure is empty.')
        return True

def remove_empty(diction):
    to_remove = list()
    #We check which ADMIN boundaries has nothing
    for dict_key in diction.keys():
    #    print(dict_key)
        if is_empty(diction[dict_key]):
            to_remove.append(dict_key)
            
    for dict_key in to_remove:
        diction.pop(dict_key)
    
    return diction

def keep_key_dict(dict_result, key):
    to_remove = []
    for dict_key in dict_result.keys():
        if key not in dict_key:
            to_remove.append(dict_key)
    for dict_key in to_remove:
        dict_result.pop(dict_key)

    return dict_result

item = 'PHL'

fn1 = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\TC_result.pkl'
fn1 = r'C:\Users\ACN980\surfdrive\Documents\Marleen\AGU2018\Dict\TC_result.pkl'
#Trying it to get it back
with open(fn1, 'rb') as f:
    TC_result = pickle.load(f)
    TC_result = remove_empty(TC_result)
    TC_result = keep_key_dict(TC_result, item)

fn2 = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\EQ_result.pkl'
fn2 = r'C:\Users\ACN980\surfdrive\Documents\Marleen\AGU2018\Dict\EQ_result.pkl'
with open(fn2, 'rb') as f:
    EQ_result = pickle.load(f)
    EQ_result = remove_empty(EQ_result)
    EQ_result = keep_key_dict(EQ_result, item)
    
fn3 = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\flood_result.pkl'
fn3 = r'C:\Users\ACN980\surfdrive\Documents\Marleen\AGU2018\Dict\flood_result.pkl'
with open(fn3, 'rb') as f:
    flood_result = pickle.load(f)
    flood_result = remove_empty(flood_result)
    flood_result = keep_key_dict(flood_result, item)
    
fn4 = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\volc_result.pkl'
fn4 = r'C:\Users\ACN980\surfdrive\Documents\Marleen\AGU2018\Dict\volc_result.pkl'
with open(fn4, 'rb') as f:
    volc_result = pickle.load(f)
    volc_result = remove_empty(volc_result)
    volc_result = keep_key_dict(volc_result, item)
    
#%%We select the admin level
admin_level = 'PHL.78.1_1'

start = '1980-01-01'
end = '2016-01-01'

#
tc = pd.DataFrame.from_dict(TC_result[admin_level], orient = 'index', dtype = str)
tc.index=pd.to_datetime(tc.index)
tc.rename(columns={0:'TC'}, inplace = True)
tc.loc[:,'TC'] = 1
tc = tc.loc[:,'TC']

flood = pd.DataFrame.from_dict(flood_result[admin_level], orient = 'index', dtype = str)
flood.index=pd.to_datetime(flood.index)
flood.rename(columns={0:'FL'}, inplace = True)
flood.loc[:,'FL'] = 1.5
flood = flood.loc[:,'FL']

eq = pd.DataFrame.from_dict(EQ_result[admin_level], orient = 'index', dtype = str)
eq.index=pd.to_datetime(eq.index)
eq.rename(columns={0:'EQ'}, inplace = True)
eq.loc[:,'EQ'] = 2
eq = eq.loc[:,'EQ']

volc = pd.DataFrame.from_dict(volc_result[admin_level], orient = 'index', dtype = str)
volc.index=pd.to_datetime(volc.index)
volc.rename(columns={0:'VO'}, inplace = True)
volc.loc[:,'VO'] =2.5
volc = volc.loc[:,'VO']

all_hz = pd.concat([tc, eq, flood, volc], axis = 1, join = 'outer')
all_hz.sort_index(inplace=True)
all_hz = all_hz.truncate(before = pd.to_datetime(start), after = pd.to_datetime(end))

#fn_out = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\Example_'+admin_level+'.csv'
#all_hz.to_csv(fn_out)

min_date = pd.to_datetime(all_hz.index.min())
max_date = all_hz.index.max()

#We plot the hazards
markersize = 50
years = mdates.YearLocator()   # every year

fn_out = r'D:\surfdrive\Documents\Marleen\AGU2018\Images\Example_Pinatubo.png'
fn_out = r'C:\Users\ACN980\surfdrive\Documents\Paper\Paper_Ruiter\Rebuttal\Example_Pinatubo.png'
f = plt.figure(figsize=(10,2))
ax = plt.axes()
#f.patch.set_visible(False)
plt.scatter(all_hz.index, all_hz['TC'], c='blue', marker="+", s=markersize) 
plt.scatter(all_hz.index, all_hz['FL'], c='lightblue', marker="+", s=markersize) #marker="|"
plt.scatter(all_hz.index, all_hz['EQ'], c='red' ,marker="+", s=markersize)
plt.scatter(all_hz.index, all_hz['VO'], c='saddlebrown', marker="+", s=markersize)
plt.ylim([0.5,3])
plt.yticks([])
#plt.xticks(pd.date_range(start='1980-01-01', end='2016-01-01', freq='AS'))
ax.xaxis.set_minor_locator(years)
#ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.show()

f.savefig(fn_out, dpi = 400)
#plt.close(f)

#labels = [item.get_text() for item in ax.get_yticklabels()]
#labels[0] = ''
#labels[1] = 'TC'
#labels[2] = 'EQ'
#labels[3] = 'FL'
#labels[4] = 'VO'
#ax.set_yticklabels(labels)
#
#f, ax = plt.subplots()
#plt.hlines(tc, tc.index, tc.index+timedelta(days=1), colors='k', linewidth=7.0, linestyles='solid')
#plt.show()
#
#oFig1 = plt.figure(1, figsize=(10,2))
#gs1 = gridspec.GridSpec(4, 4)
#gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
#
#oFig1.add_subplot(4,1,1) 
#plt.scatter(all_hz.index, all_hz['TC'], c='lawngreen', marker="s", s=markersize)
#oFig1.add_subplot(4,1,2) 
#plt.scatter(all_hz.index, all_hz['FL'], c='blue', marker="s", s=markersize)
#oFig1.add_subplot(4,1,3) 
#plt.scatter(all_hz.index, all_hz['EQ'], c='red' ,marker="s", s=markersize)
#plt.show()




























