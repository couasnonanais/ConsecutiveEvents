# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:05:26 2019

@author: Anton
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:13:03 2018

@author: Anton
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from datetime import datetime  
from datetime import timedelta 
from collections import Counter, defaultdict
import collections
import itertools
import matplotlib.gridspec as gridspec



#Plot time series with the disasters - selecting location of Pinatubo
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) #see: https://matplotlib.org/examples/color/named_colors.html

def create_event_start(dict_hz, admin_level, haz_type, ID_int):
    tc = pd.DataFrame.from_dict(dict_hz[admin_level], orient = 'index', dtype = str)
    tc.sort_index(inplace=True)
    tc.index=pd.to_datetime(tc.index)
    tc.rename(columns={0:haz_type}, inplace = True)
    tc.drop_duplicates(subset=haz_type, keep='first', inplace=True) #We only keep starting dates
    tc.loc[:,haz_type] = ID_int
    tc = tc.loc[:,haz_type]
    return tc

def create_seq_hazard(all_hz):
    dummy = all_hz.iloc[:,0]
    for j in range(all_hz.shape[1]-1):
        next_col = all_hz.columns[j+1]
        dummy.fillna(value = all_hz.loc[:,next_col], inplace = True)        
    return dummy

def transition_matrix(transitions):   
    n = 1+ max(transitions) #number of states
    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def inter_arrival_pdf(series_all_hz, transitions_name):
    all_data = pd.DataFrame(series_all_hz, dtype = int)
    all_data.reset_index(inplace=True)
    all_data.rename(columns={'index':'date'}, inplace = True)
    col_name = all_data.columns[-1]
        
    cat_states = {}
    for i in transitions_name:
        cat_states[i] = dict.fromkeys(transitions_name)
    for i in transitions_name:
        for j in transitions_name:
            cat_states[i][j] = list()
    
    for (i,j) in zip(all_data[col_name].index,all_data[col_name][1:].index):
        inter_time = all_data['date'][j]-all_data['date'][i]
        cat_states[all_data[col_name][i]][all_data[col_name][j]].append(inter_time.days)
    return cat_states

def event_duration_pdf(dict_hz, admin_level, haz_type):
    tc =  pd.DataFrame.from_dict(dict_hz[admin_level], orient = 'index', dtype = str)
    tc.sort_index(inplace=True)
    tc.index=pd.to_datetime(tc.index)
    tc.rename(columns={0:haz_type}, inplace = True)    
    TC_counts = Counter(tc[haz_type])
    df = pd.DataFrame.from_dict(TC_counts, orient='index')    
    #Store the duration
    res = {haz_type:np.array(df.iloc[:,0])}
    return res

def add_duration(haz, per_haz, all_dur):
# We add a duration
    dum = per_haz[haz].dropna()    
    final_haz= pd.DataFrame(index=np.arange(min(per_haz.index), max(per_haz.index)+1000, 1), columns = [haz])
    last_day_haz = pd.DataFrame(index=np.arange(min(per_haz.index), max(per_haz.index)+1000, 1), columns = [haz])
    fin_haz_id = pd.DataFrame(index=np.arange(min(per_haz.index), max(per_haz.index)+1000, 1), columns = [haz])
    k = 0
    for i in dum.index:
        haz_id = str(haz)+"_"+str(k)
        dur_i = np.random.choice(all_dur[haz], replace = True) 
        final_haz.loc[i:(i+dur_i),haz] = haz
        fin_haz_id.loc[i:(i+dur_i),haz] = haz_id
        last_day_haz.loc[i+dur_i,haz] = haz
        k+=1
        
    return final_haz, fin_haz_id, last_day_haz

def cond_prob_type(time_window, all_fin, all_fin_id, result, haz_ID):
        new_index = np.arange(min(all_fin.index), max(all_fin.index)+1, 1)
        all_day_fin = pd.DataFrame(index = new_index)
        all_day_fin = pd.concat([all_day_fin,all_fin_id], axis = 1)
    
        sel_result_index = result[result==haz_ID].dropna().index
        nb_day_haz = pd.DataFrame(index = sel_result_index, columns=[time_window]) #nb_day_haz = pd.DataFrame(index = fin_dates.index, columns=[time_window])
        direc = 1
        for j in time_window:
            for i in sel_result_index: #    for i in fin_dates.index:
#                print(i)
                try:
                    check = list(pd.unique(all_day_fin.loc[i:(i+(direc*j)), :].values.ravel('K'))) #pd.unique(all_day_fin.loc[(i+1):(i+(direc*j)), :].values.ravel('K'))
                    check.remove(np.nan)  
                    nb_day_haz.loc[i, j] = len(check)-1 ###If starting from beginning. OTHERWISE len(check) only
                except:
                    nb_day_haz.loc[i, j] = np.nan
        # We calculate the probability
        nb_events = [1,2,3,4,5]
        all_prob = pd.DataFrame(index = nb_events, columns = [time_window])
        for i in time_window:
#            print(i)
            to_drop = nb_day_haz.index[nb_day_haz.index >= max(nb_day_haz.index)-i]
            sel = nb_day_haz[i].drop(labels = to_drop, axis = 0)
            for j in nb_events:
        #        print(i,j)
                all_prob.loc[j,i]=len(sel[sel>=j].dropna())/len(sel)
        return all_prob

def make_checker_figure(time_window, all_prob, fn, save_fig):
    import seaborn as sns
    col_fig =[ str(i)  for i in time_window] #np.arange(0,len(time_window),1)
    df_fig = pd.DataFrame(data=np.array(all_prob),index = all_prob.index, columns=col_fig) #['lag_0','lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    df_fig = df_fig[df_fig.columns].astype(float) 
   
    fig, ax = plt.subplots()
    cmap = plt.cm.Reds 
    bounds = np.linspace(0, 1 ,11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sns.heatmap(df_fig, annot=True, fmt='.2f', cmap=cmap, norm=norm, ax=ax, linewidths=0.05, linecolor='k') #, cbar_kws=dict(ticks=[-1, -0.5, -0.3, -0.1, +0.1, +0.3, +0.5, +1]))
    ax.invert_yaxis()
    plt.xlabel('Time Window (days)')
    plt.ylabel('Prob. of at least X event occuring after one event')
    plt.show()
    if save_fig == True:
        fig.savefig(fn, dpi=300)
    
    return

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
#Trying it to get it back
with open(fn1, 'rb') as f:
    TC_result = pickle.load(f)
    TC_result = remove_empty(TC_result)
    TC_result = keep_key_dict(TC_result, item)

fn2 = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\EQ_result.pkl'
with open(fn2, 'rb') as f:
    EQ_result = pickle.load(f)
    EQ_result = remove_empty(EQ_result)
    EQ_result = keep_key_dict(EQ_result, item)
    
fn3 = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\flood_result.pkl'
with open(fn3, 'rb') as f:
    flood_result = pickle.load(f)
    flood_result = remove_empty(flood_result)
    flood_result = keep_key_dict(flood_result, item)
    
fn4 = r'D:\surfdrive\Documents\Marleen\AGU2018\Dict\volc_result.pkl'
with open(fn4, 'rb') as f:
    volc_result = pickle.load(f)
    volc_result = remove_empty(volc_result)
    volc_result = keep_key_dict(volc_result, item)
    
#%%We select the admin level
admin_level = 'PHL.78.1_1'
start = '1980-01-01'
end = '2016-01-01'
nb_days = pd.to_datetime(end)-pd.to_datetime(start)

fn_out = 'D:\surfdrive\Documents\VU\Documents\EGU2019\Time_series_start'

tc = create_event_start(TC_result, admin_level, 'TC', 0)
tc_dur = event_duration_pdf(TC_result, admin_level, 'TC')
tc_dur[0] = tc_dur.pop('TC')
tc.to_csv(os.path.join(fn_out,"TC.csv"))

flood = create_event_start(flood_result, admin_level, 'FL', 1)
flood_dur = event_duration_pdf(flood_result, admin_level, 'FL')
flood_dur[1] = flood_dur.pop('FL')
flood.to_csv(os.path.join(fn_out,"FL.csv"))

eq = create_event_start(EQ_result, admin_level, 'EQ', 2)
eq_dur = event_duration_pdf(EQ_result, admin_level, 'EQ')
eq_dur[2] = eq_dur.pop('EQ')
eq.to_csv(os.path.join(fn_out,"EQ.csv"))

volc = create_event_start(volc_result, admin_level, 'VO', 3)

volc_dur = event_duration_pdf(volc_result, admin_level, 'VO')
volc_dur[3] = volc_dur.pop('VO')

all_dur = tc_dur
for d in [flood_dur, eq_dur, volc_dur]:
    all_dur.update(d)
#%%  
all_hz = pd.concat([tc, flood, eq, volc], axis = 1, join = 'outer')
all_hz.sort_index(inplace=True)
all_hz = all_hz.truncate(before = pd.to_datetime(start), after = pd.to_datetime(end))
all_hz.loc['1991-06-15','TC'] = np.nan
all_hz.loc['1991-06-16',"TC"] = 0

transitions_name = [0,1,2,3]
series_all_hz = create_seq_hazard(all_hz)
transition_prob = transition_matrix(np.array(series_all_hz, dtype = int))
all_inter = inter_arrival_pdf(series_all_hz, transitions_name)

#%% We create a time series of hazard sequence
start_hazard = 0
list_hazards =  np.array(start_hazard)

next_hazard = np.random.choice(transitions_name,replace=True,p=transition_prob[start_hazard])
np.append(list_hazards, next_hazard)
total_steps = 3000

for i in range(total_steps-1):
    prev_haz = next_hazard
    next_hazard = np.random.choice(transitions_name,replace=True,p=transition_prob[prev_haz])
    list_hazards = np.append(list_hazards, next_hazard)

#%% We add some interarrival time in between 
result = pd.DataFrame(columns=['sim'])
k=0
for (i,j) in zip(list_hazards, list_hazards[1:]):
#    print(i,j)
    result.loc[k, 'sim']=i
    inter = np.random.choice(all_inter[i][j], replace = True)
    result.loc[k+inter, 'sim'] = j
    k = k+inter
#%% We separate per hazard and add duration
per_haz = pd.DataFrame()
for haz in transitions_name:
#    print(haz)
    dum = result == haz
    per_haz = pd.concat([per_haz, result[dum].rename(columns={'sim':haz})], axis =1)    

tc_fin, tc_fin_id, tc_last_day = add_duration(0, per_haz, all_dur)
flood_fin, flood_fin_id, flood_last_day = add_duration(1, per_haz, all_dur)
eq_fin, eq_fin_id, eq_last_day = add_duration(2, per_haz, all_dur)    
volc_fin, volc_fin_id, volc_last_day = add_duration(3, per_haz, all_dur)

all_fin = pd.concat([tc_fin, flood_fin, eq_fin, volc_fin], axis = 1).dropna(axis = 0, how = 'all')
all_fin_id = pd.concat([tc_fin_id, flood_fin_id, eq_fin_id, volc_fin_id], axis = 1).dropna(axis = 0, how = 'all')
all_last_day = pd.concat([tc_last_day, flood_last_day, eq_last_day, volc_last_day], axis = 1).dropna(axis = 0, how = 'all')
fin_dates = create_seq_hazard(all_last_day)
#%%
markersize = 22

f = plt.figure(figsize=(10,2))
f.patch.set_visible(False)
plt.scatter(all_fin.index, all_fin.loc[:,0] + 1, c='purple', marker="|", s=markersize) 
plt.scatter(all_fin.index, all_fin.loc[:,1] + 0.5, c='blue', marker="|", s=markersize)
plt.scatter(all_fin.index, all_fin.loc[:,2], c='red' ,marker="|", s=markersize)
plt.scatter(all_fin.index, all_fin.loc[:,3] - 0.5, c='green', marker="|", s=markersize)
plt.ylim([0.5,3.5])
plt.yticks([])
plt.show()

nb_periods = int(all_fin.index[-1]/nb_days.days)

nb_fig = 10
seq_i = np.random.choice(np.arange(0, nb_periods, 1), size = (nb_fig,1), replace = False)
fn_out = r'D:\surfdrive\Documents\VU\Documents\EGU2019\Figures'
for i in np.arange(0,nb_fig,1):

    random_window = seq_i[i]*nb_days.days
    #random_window = np.random.choice(np.arange(min(all_fin.index)+nb_days.days, max(all_fin.index)-nb_days.days, 1))
    fig_path = 'random_'+str(i)+'.png'    

    f = plt.figure(figsize=(10,2))
    f.patch.set_visible(False)
    plt.scatter(all_fin.index, all_fin.loc[:,0] + 1, c='purple', marker="|", s=markersize) 
    plt.scatter(all_fin.index, all_fin.loc[:,1] + 0.5, c='blue', marker="|", s=markersize)
    plt.scatter(all_fin.index, all_fin.loc[:,2], c='red' ,marker="|", s=markersize)
    plt.scatter(all_fin.index, all_fin.loc[:,3] - 0.5, c='green', marker="|", s=markersize)
    plt.ylim([0.5,3.5])
    plt.yticks([])
    plt.xlim([random_window, random_window+nb_days.days])
    plt.show()

    f.savefig(os.path.join(fn_out,fig_path), dpi = 400)
    plt.close(f)

#%% We calculate the number of events
time_window = [3,15,30,90,180,365]
new_index = np.arange(min(all_fin.index), max(all_fin.index)+1, 1)
all_day_fin = pd.DataFrame(index = new_index)
all_day_fin = pd.concat([all_day_fin,all_fin_id], axis = 1)

nb_day_haz = pd.DataFrame(index = result.index, columns=[time_window]) #nb_day_haz = pd.DataFrame(index = fin_dates.index, columns=[time_window])
direc = 1
for j in time_window:
    for i in result.index: #    for i in fin_dates.index:
        try:
            check = list(pd.unique(all_day_fin.loc[i:(i+(direc*j)), :].values.ravel('K'))) #pd.unique(all_day_fin.loc[(i+1):(i+(direc*j)), :].values.ravel('K'))
            check.remove(np.nan)  
            nb_day_haz.loc[i, j] = len(check)-1 ###If starting from beginning. OTHERWISE len(check) only
        except:
            nb_day_haz.loc[i, j] = np.nan

# We calculate the probability
nb_events = [1,2,3,4,5]
all_prob = pd.DataFrame(index = nb_events, columns = [time_window])

for i in time_window:
#    print(i)
    to_drop = nb_day_haz.index[nb_day_haz.index >= max(nb_day_haz.index)-i]
    sel = nb_day_haz[i].drop(labels = to_drop, axis = 0)
    for j in nb_events:
#        print(i,j)
        all_prob.loc[j,i]=len(sel[sel>=j].dropna())/len(sel)

all_prob_typh = cond_prob_type(time_window, all_fin, all_fin_id, result, 0)
all_prob_fl = cond_prob_type(time_window, all_fin, all_fin_id, result, 1)
all_prob_eq = cond_prob_type(time_window, all_fin, all_fin_id, result, 2)
all_prob_volc = cond_prob_type(time_window, all_fin, all_fin_id, result, 3)


#%% Figure 
f = plt.figure(figsize=(6,6))
for i in all_prob.index:
#    print(i)
    plt.plot(np.arange(0,6,1), all_prob.loc[i,:], '-', label = str(i)+' events or more')
plt.legend()
plt.ylim([-0.05,1.05])

#%%
fn_base = r'D:\surfdrive\Documents\VU\Documents\EGU2019\Figures'
fn = os.path.join(fn_base,'Example_markov_sns.png')
make_checker_figure(time_window, all_prob, fn, save_fig=True)

fn = os.path.join(fn_base,'Example_markov_typhoon.png')
make_checker_figure(time_window, all_prob_typh, fn, save_fig=True)

fn = os.path.join(fn_base,'Example_markov_flood.png')
make_checker_figure(time_window, all_prob_fl, fn, save_fig=True)

fn = os.path.join(fn_base,'Example_markov_eq.png')
make_checker_figure(time_window, all_prob_eq, fn, save_fig=True)

fn = os.path.join(fn_base,'Example_markov_volc.png')
make_checker_figure(time_window, all_prob_volc, fn, save_fig=True)

        









