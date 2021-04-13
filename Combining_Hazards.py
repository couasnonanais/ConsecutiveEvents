# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:34:21 2018

@author: Anton
"""
import pickle
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime , timedelta
import calendar
import matplotlib.pyplot as plt
import os
#Once all the hazards are done, we will look per admin level

def is_empty(any_structure):
    if any_structure:
        #print('Structure is not empty.')
        return False
    else:
        #print('Structure is empty.')
        return True

def calc_mono_cond_prob_before(haz, haz_name, delta_days = 3): #Case could be after
    unique_dates = haz.drop_duplicates(subset = [haz_name], keep = 'first', inplace = False)
    comp = haz.drop_duplicates(subset = [haz_name], keep = 'last', inplace = False)
    mult = - 1
    
    for i in unique_dates.index:
        #print(i)
        haz_ID = unique_dates.loc[i, haz_name]        
        rem = comp[haz_name].where(comp[haz_name]==haz_ID).dropna().index    
        
        find_date = unique_dates.loc[i, 'date'] + mult * timedelta(days = delta_days)
        check = pd.date_range(find_date, unique_dates.loc[i, 'date'] - timedelta(days = 0)) #######
        
        if any(comp.drop(rem,axis=0).date.isin(check)):
            if np.sum(comp.drop(rem,axis=0).date.isin(check))> 0:
                unique_dates.loc[i,'hit'] = 1
        else:
            unique_dates.loc[i,'hit'] = 0
    
    cond_prob = unique_dates.loc[:,'hit'].sum() / unique_dates.shape[0]
    nb_events = unique_dates.shape[0]        
    return (cond_prob, nb_events)

def calc_mono_cond_prob_after(haz, haz_name, delta_days = 3): #Case could be after
    unique_dates = haz.drop_duplicates(subset = [haz_name], keep = 'last', inplace = False)
    comp = haz.drop_duplicates(subset = [haz_name], keep = 'first', inplace = False)
    mult = 1

    for i in unique_dates.index:
        #print(i)
        haz_ID = unique_dates.loc[i, haz_name]        
        rem = comp[haz_name].where(comp[haz_name]==haz_ID).dropna().index
                
        find_date = unique_dates.loc[i, 'date'] + mult * timedelta(days = delta_days)
        check = pd.date_range(unique_dates.loc[i, 'date'] + timedelta(days = 0), find_date) 
        
        if any(comp.drop(rem,axis=0).date.isin(check)):
            if np.sum(comp.drop(rem,axis=0).date.isin(check))> 0:
                unique_dates.loc[i,'hit'] = 1
        else:
            unique_dates.loc[i,'hit'] = 0
    
    cond_prob = unique_dates.loc[:,'hit'].sum() / unique_dates.shape[0]
    nb_events = unique_dates.shape[0]        
    return (cond_prob, nb_events)

def calc_interarrival_bw_events_start():
    return

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

def sel_area(diction, start_dict_keys = 'PHL'):
    to_keep = list()
    
    for dict_key in diction.keys():
        if start_dict_keys in dict_key:
            to_keep.append(dict_key)

    dict_you_want = { your_key: diction[your_key] for your_key in to_keep }
    return dict_you_want

def sel_time(diction, year_start = pd.to_datetime('1960-01-01'), year_end = pd.to_datetime('2016-01-01')):
  
    data = pd.DataFrame.from_dict(diction, orient = 'columns')
    data.reset_index(inplace = True)
    data.rename({'index':'date'}, axis = 'columns', inplace = True)
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date',axis = 0, inplace = True)
    data.set_index('date', inplace = True)
    
    print('First entry: ', data.index.min())
    print('Last entry: ', data.index.max())
    data_truncated = data.truncate(before = year_start, after = year_end)
    
    return data_truncated

def admin_level_areas(dataframe, start_dict_keys = 'PHL'):
    to_keep = list()
    for value in dataframe['GID_2']:
      if start_dict_keys in value:
            to_keep.append(value)  
    
    return to_keep

def selec_admin_hazard(dataframe, admin, name):
    try:
        tc = pd.Series(dataframe.loc[:,admin], name = name)
        tc.dropna(inplace = True)
    except:
        tc = pd.Series(np.nan)
    return tc
#%%
#1-We call one hazard
fn_out1 = r'F:\CONSECUTIVE\MODELING\DATA_INPUT\TC\1960\Dict\TC_result.pkl'
with open(fn_out1, 'rb') as f:
    TC_result = pickle.load(f)
    TC_result = sel_area(TC_result, "PHL")
    TC_result = remove_empty(TC_result)

fn_2 = r'F:\CONSECUTIVE\MODELING\DATA_INPUT\EQ\Dict\EQ_result.pkl'
with open(fn_2, 'rb') as f:
    EQ_result = pickle.load(f)
    EQ_result = sel_area(EQ_result, "PHL")
    EQ_result = remove_empty(EQ_result)
    
fn_3 = r'F:\CONSECUTIVE\MODELING\DATA_INPUT\FLOOD\1980\Dict\flood_result.pkl'
with open(fn_3, 'rb') as f:
    flood_result = pickle.load(f)
    flood_result = sel_area(flood_result, "PHL")
    flood_result = remove_empty(flood_result)

fn_4 = r'F:\CONSECUTIVE\MODELING\DATA_INPUT\VOLCANO\1900\Dict\volc_result.pkl' 
with open(fn_4, 'rb') as f:
    volc_result = pickle.load(f)
    volc_result = sel_area(volc_result, "PHL")
    volc_result = remove_empty(volc_result)

#%% We make it into dataframe from 1980 - 2016
#TC = sel_time(TC_result, year_start = pd.to_datetime('1980-01-01'), year_end = pd.to_datetime('2016-01-01'))
#EQ = sel_time(EQ_result, year_start = pd.to_datetime('1980-01-01'), year_end = pd.to_datetime('2016-01-01'))
#Volc = sel_time(volc_result, year_start = pd.to_datetime('1980-01-01'), year_end = pd.to_datetime('2016-01-01'))
#Flood = sel_time(flood_result, year_start = pd.to_datetime('1980-01-01'), year_end = pd.to_datetime('2016-01-01'))

#%%
#fn_admin = r'P:\Marleen\AdministrativeBoundaries\Project\gadm36_2_1.shp'
fn_admin = r'F:\CONSECUTIVE\MODELING\DATA_INPUT\ADMIN_LEVELS\gadm36_2_1.shp'
admin = gpd.read_file(fn_admin)
all_admin_PHL = admin_level_areas(admin, start_dict_keys = 'PHL')

##%% We start the calculation for all admin levels
#for adm in all_admin_PHL:
#    print(adm)
#    #We select the admin level in the data
#    tc = selec_admin_hazard(TC, adm, 'TC')
#    eq = selec_admin_hazard(EQ, adm, 'EQ')
#    flood = selec_admin_hazard(Flood, adm, 'Flood')
#    volc = selec_admin_hazard(Volc, adm, 'VOL')
    
#%%  Checking where both items are present
#Intersection of ADMIN levels
intersect = []
for item in TC_result.keys():
    if item in EQ_result:
        intersect.append(item)
        print("Intersects:", item)

subset = admin[admin.GID_2.isin(intersect)] #Multi - Intersect is where both hazards are present
        
#%% We plot those regions
haz_type = {'TC': list(TC_result.keys()) , 'EQ': list(EQ_result.keys()), 
            'Flood': list(flood_result.keys()), 'Volc':list(volc_result.keys())} #Select mono hazard

haz_name = 'Volc'

if haz_name == 'TC':
    haz_result = TC_result.copy()
    mono = admin[admin.GID_2.isin(haz_type['TC'])]
elif haz_name == 'EQ':
    haz_result = EQ_result.copy()
    mono = admin[admin.GID_2.isin(haz_type['EQ'])]
elif haz_name == 'Flood':
    haz_result = flood_result.copy()
    mono = admin[admin.GID_2.isin(haz_type['Flood'])]
else:
#    haz_name = 'Volc'
    haz_result = volc_result.copy()
    mono = admin[admin.GID_2.isin(haz_type['Volc'])]

#%% We select one region and look at 1- the interarrival rate 2- Probabilities 

#MONO-HAZARD
test = mono.loc[:,['GID_2']] #All the Admin level indicated  
test.reset_index(inplace = True, drop = True)

time_window = 3 #Time window of X days before #Also did 3 days before

start = '1980-01-01'
end = '2016-01-01'

for item in test.index: #We calculate per admin level (could be any scope really) 
#Make read TC first
#2-10-2018: all TC, 3 days - Stopped at 8626 --> 8627 is not done
    print(item) #PUT COLUMN NAME
    haz = pd.DataFrame.from_dict(haz_result[test.loc[item,'GID_2']], orient='index',dtype = str) #For mono this step is not necessary but for multiple yes
    haz.rename(columns={0:haz_name}, inplace = True)
    haz.index = pd.to_datetime(haz.index)  
    haz.sort_index(inplace = True)     
    haz = haz.truncate(before = pd.to_datetime(start), after = pd.to_datetime(end))
    haz.reset_index(inplace = True, drop = False)
    haz.rename(columns={'index':'date'},inplace = True)
    haz.sort_values('date',axis = 0, inplace = True)    
    
    try:
        (cond_prob, nb_events) = calc_mono_cond_prob_before(haz, haz_name, delta_days = time_window)
      
        test.loc[item, 'cond_prob'] = cond_prob
        test.loc[item, 'nb_events'] = nb_events 
    except:
        continue

test.dropna(inplace=True)

#Storing the results
file_out='result.csv'
#file_out='result_suite.csv'
fn_time = '{0:1.0f}_days'.format(time_window)
fn_folder = os.path.join('F:\CONSECUTIVE\MODELING\OUTPUT\MONO\Y1980_2016', haz_name, fn_time, file_out)
test.to_csv(fn_folder, index_label = 'index')

##Making a histogram
#bins = np.linspace(0, 1, 100)
#plt.figure()
#plt.hist(test['cond_prob'], bins, facecolor='g', alpha=0.5) 
#plt.xlabel('Conditional Probability: P(EQ$_{t-n}$|EQ)')
#plt.ylabel('Frequency')
#plt.show()

#%% PLOTTING MONO-HAZARS: intersect
time_window = [3,30]
haz_name = 'EQ'
file_out='result.csv'

#fn_time = '{0:1.0f}_days'.format(time_window[0])
#haz_dt1 = os.path.join('F:\CONSECUTIVE\MODELING\OUTPUT\MONO\Subset_EQ_intersect_TC', haz_name, fn_time, file_out)
#fn_time = '{0:1.0f}_days'.format(time_window[1])
#haz_dt2 = os.path.join('F:\CONSECUTIVE\MODELING\OUTPUT\MONO\Subset_EQ_intersect_TC', haz_name, fn_time, file_out)

fn_time = '{0:1.0f}_days'.format(time_window[0])
haz_dt1 = os.path.join('F:\CONSECUTIVE\MODELING\OUTPUT\MULTI\EQ_TC', fn_time, file_out)
fn_time = '{0:1.0f}_days'.format(time_window[1])
haz_dt2 = os.path.join('F:\CONSECUTIVE\MODELING\OUTPUT\MULTI\EQ_TC', fn_time, file_out)

test_dt1 = pd.read_csv(haz_dt1, index_col = 'index')
test_dt2 = pd.read_csv(haz_dt2, index_col = 'index')

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
plt.xlabel('Conditional Probability: P(H$_{t-n}$|H)')
plt.ylabel('Frequency')

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.5, 0.4, 0.3, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])

ax2.hist(test_dt2['cond_prob'], bins, facecolor='g', alpha=0.5) 
ax2.hist(test_dt1['cond_prob'], bins, facecolor='b', alpha=0.5)
ax2.axis([0.01, 0.55, 0, 100])

plt.show()
#fig.savefig(r'F:\CONSECUTIVE\MODELING\OUTPUT\MONO\EQ\3_30days_intersectEQ.png', bbox_inches = 'tight', dpi = 300)
fig.savefig(r'F:\CONSECUTIVE\MODELING\OUTPUT\MULTI\EQ_TC\3_30days_intersectEQ.png', bbox_inches = 'tight', dpi = 300)
   

#%% MULTI_HAZARD COND_PROB
def hazard_rename(haz,haz_name, haz_name_out = 'dis_type'):
     dummy = haz.loc[:,['date',haz_name]].copy()
     dummy.rename(columns={haz_name: haz_name_out}, inplace = True)
     dummy.dropna(axis=0, inplace = True)
     return dummy

def make_multi_to_mono(haz, date_label = 'date'): #Case could be after
    haz = sel.copy()
    haz_columns_name = haz.columns.drop(date_label)
    
    dummy1 = hazard_rename(haz, haz_columns_name[0], 'dis_type')
    
    for colname in haz_columns_name[1:]:
        #print(colname)
        dummy2 = hazard_rename(haz, colname, 'dis_type')
        dummy1 = pd.concat([dummy1,dummy2], axis = 0)

    dummy1.sort_values('date',axis = 0, inplace = True)    
    return dummy1
    
#We select the first one:
test = subset.loc[subset.index,['GID_2']] #subset.index[0:100]
test.reset_index(inplace = True, drop = True)

time_window = 3

for item in test.index:
    print(item) #PUT COLUMNA NAME
    tc = pd.DataFrame.from_dict(TC_result[test.loc[item,'GID_2']], orient='index',dtype = str)
    tc.rename(columns={0:"TC"}, inplace = True)
    tc.index = pd.to_datetime(tc.index)
    eq = pd.DataFrame.from_dict(EQ_result[test.loc[item,'GID_2']], orient='index',dtype = str)
    eq.rename(columns={0:"EQ"}, inplace = True)    
    eq.index = pd.to_datetime(eq.index)
    
    sel = pd.concat([tc, eq], axis = 0, join = 'outer')
    sel.reset_index(inplace = True, drop = False)
    sel.rename(columns={'index':'date'},inplace = True)
    sel.sort_values('date',axis = 0, inplace = True)
    sel.set_index('date', inplace = True)
    sel = sel.truncate(before = pd.to_datetime(start), after = pd.to_datetime(end))
    sel.reset_index(inplace = True, drop = False)
    
    try:    
        haz_all = make_multi_to_mono(sel, date_label = 'date')
        haz_all.reset_index(inplace = True, drop = True)
        (cond_prob, nb_events) = calc_mono_cond_prob_before(haz_all, 'dis_type', delta_days = time_window)
        test.loc[item, 'cond_prob'] = cond_prob
        test.loc[item, 'nb_events'] = nb_events 
    except:
        continue

#Storing the results
file_out='result.csv'
fn_time = '{0:1.0f}_days'.format(time_window)
fn_folder = os.path.join('F:\CONSECUTIVE\MODELING\OUTPUT\MULTI', 'Y1980_2016','EQ_TC', fn_time, file_out)
test.to_csv(fn_folder, index_label = 'index')

#%% Calculate probability
#If you have a TC, what is the chance of observing a EQ?

def cond_prob_hit(tc, eq, time_window):
    tc_sel = tc.drop_duplicates(subset = ['TC'], keep = 'first', inplace = False)
    eq_sel = eq.copy()
    eq_sel.reset_index(inplace = True, drop = False)
    eq_sel.rename(columns={'index':"Hazard"}, inplace = True)
    
    for date_1 in tc_sel.index:
        #print(date_1)
        #PBM WITH CHECK --> IF IT IS THE SAME HAZARD THEN IT WILL COUNT THE HAZARD AS WELL...
        check = pd.date_range(date_1 - timedelta(days = time_window), date_1 - timedelta(days = 1)) #CHECK THIS
        eq_sel.Hazard.isin(check)
        if any(eq_sel.Hazard.isin(check)):
            tc_sel.loc[date_1,'hit'] = 1
        else:
            tc_sel.loc[date_1,'hit'] = 0
            
    prob_calc = tc_sel.hit.sum()/tc_sel.hit.size
    print(prob_calc)
    return prob_calc 
       
    
    plt.figure()
    plt.hist(inter_days, 100, facecolor='g', alpha=0.5)
    plt.xlabel('Inter-arrival time between start of events (days)')
    plt.ylabel('Nb. of events')
    plt.show()
    
    inter_days = inter_arr_start.days
    
    #Make a function for interarrival time
    #Make a beginning and end columsn
    #Calculate interarrival time
    # 1- Drop duplicates per 
    





























