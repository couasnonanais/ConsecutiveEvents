# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:48:18 2019

@author: ACN980
"""

import pickle
import json
from datetime import datetime, date
import os

#%%
def date_handler(obj, datetime=datetime, date=date):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    print(obj, type(obj))
    raise TypeError("Type not serializable")

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

    
#%%
d = {'test': datetime(2018, 1, 1, 1, 1, 1)}

with open('test.json', 'w') as f:
    json.dump(d, f, default=date_handler)
#%%
fn1 = r'D:\surfdrive\Documents\Marleen\JSON\Dicts\TC_result.pkl'

with open(fn1, 'rb') as f:
    TC_result = pickle.load(f)
    TC_result = remove_empty(TC_result)

fn2 = r'D:\surfdrive\Documents\Marleen\JSON\Dicts\EQ_result.pkl'
with open(fn2, 'rb') as f:
    EQ_result = pickle.load(f)
    EQ_result = remove_empty(EQ_result)

fn_out = r'D:\surfdrive\Documents\Marleen\JSON'
with open(os.path.join(fn_out,'TC_admin2_1960_2016.json'), 'w') as f1:
    json.dump(TC_result, f1)

with open(os.path.join(fn_out,'EQ_admin2_1960_2016.json'), 'w') as f2:
    json.dump(EQ_result, f2)


#Testing if it works
with open(os.path.join(fn_out,'TC_admin2_1960_2016.json')) as fh:
    a = json.load(fh)
with open(os.path.join(fn_out,'EQ_admin2_1960_2016.json')) as fh:
    b = json.load(fh)









