# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:44:22 2018

@author: pc
"""

import os
import numpy as np 
import pandas as pd 
from scipy import stats
sub_path = "Submit/newmethod"
all_files = os.listdir(sub_path)
all_files.remove('best_submit.csv')

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: all_files[x], range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()

corr = concat_sub.corr()

best_case = pd.read_csv('Submit/Blend/bestsubmit.csv')

#del concat_sub['Bagging1302.csv']
# get the data fields ready for stacking
concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:8].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:8].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:8].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:8].median(axis=1)
concat_sub['log_mean'] = np.exp(np.mean(concat_sub.iloc[:,1:8].apply(lambda x: np.log(x)), axis=1))
concat_sub['trim_mean'] = stats.trim_mean(concat_sub.iloc[:, 1:8], 0.15, axis=1)


concat_sub['weight_min_trim_mean'] = 0.6*concat_sub['is_iceberg_min'] + 0.4*concat_sub['trim_mean']
concat_sub['weight_max_trim_mean'] = 0.6*concat_sub['is_iceberg_max'] + 0.4*concat_sub['trim_mean']

concat_sub['weight_trim_mean'] = best_case['is_iceberg']*0.5 + 0.5 * concat_sub['trim_mean']
concat_sub['weight_mean'] = best_case['is_iceberg']*0.3 + 0.7 * concat_sub['is_iceberg_mean']
concat_sub['weight_median'] = best_case['is_iceberg']*0.2 + 0.8 * concat_sub['is_iceberg_median'] 

concat_sub['score'] = 5*best_case['is_iceberg'] +(3*concat_sub['trim_mean']+3*concat_sub['log_mean']+
          3*concat_sub['is_iceberg_median']+3*concat_sub['is_iceberg_mean']) + concat_sub.iloc[:, 1:8].sum(axis=1)
concat_sub['score'] = concat_sub['score']/(5+3*4+7)


concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.00001, 0.99999)

All = concat_sub

All['is_iceberg'] = np.where(All['score']<0.15, All['is_iceberg_min'], 
                                     np.where(All['score']>0.85, All['is_iceberg_max'], 
                                     np.where(np.logical_and(All['score']>=0.1, All['score']<0.3), 
                                              All['weight_min_trim_mean'], np.where(np.logical_and(All['score']>0.7, 
                                              All['score']<=0.85), All['weight_max_trim_mean'], 
                                              np.where(np.logical_and(All['score']>=0.3, All['score']<0.5),
                                              All['weight_mean'], All['weight_median'])))))
new = pd.DataFrame()
new ['id'] = All['id']
new['is_iceberg'] = All['is_iceberg']
new['is_iceberg'] = np.clip(new['is_iceberg'].values, 0.0001, 0.9999)
new.to_csv('bagging_test_.csv',  index=False)

