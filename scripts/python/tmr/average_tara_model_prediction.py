# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:18:22 2020

@author: drami
"""
import pandas as pd
from keras.models import load_model
import math
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(1,'scripts/python/tmr/')
from cnn_functions import seq_one_hot
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error

regress_var='T_C'
model_save_path='data/models/tara_T_regression.h5'
data_path='data/tara/tara_merged_all_data.csv'
n=1

#nn parameters
max_len=500
cnn_fun_path='scripts/python/tmr/'
seq_type='aa'
seq_resize=False




model=load_model(model_save_path)
 
seq_df=pd.read_csv(data_path)

seq_df=seq_df[['sequence','Site','T_C']]
seq_df=seq_df.groupby('Site').sample(frac=0.05)
# del seq_df

# batch_size=10000

# def parallel_predict_tara(seq_df,i,batch_size,model):
#     max_n=math.floor(seq_df.shape[0]/batch_size)
#     print(i)
#     min_i=i*max_n
#     if i<batch_size-1:
#         max_i=(i+1)*max_n-1
#     else:
#         max_i=seq_df.shape[0]-1
            
#     sub=seq_df.iloc[min_i:max_i]
#     test_one_hot=seq_one_hot(sub['sequence'],
#                                   seq_type=seq_type,
#                                   max_len=max_len,
#                                   seq_resize=seq_resize)
#     tmp=model.predict(test_one_hot)
#     size_reshape=tmp.shape[0]
#     sub['prediction']=tmp.reshape(-1,size_reshape)[0]
#     sub.to_csv('data/tara/tara_predict_parallel/subsample_'+str(i)+"_"+str(i+1)+".csv")

# ytest=np.array(test[regress_var],dtype=float)

# Parallel(n_jobs=n)(delayed(parallel_predict_tara)(seq_df,i,batch_size,model) for i in range(batch_size))
test_one_hot=seq_one_hot(seq_df['sequence'],
                              seq_type=seq_type,
                              max_len=max_len,
                              seq_resize=seq_resize)

tmp=model.predict(test_one_hot)
size_reshape=tmp.shape[0]
seq_df['prediction']=tmp.reshape(-1,size_reshape)[0]
seq_df_mean=seq_df.groupby('Site')['prediction','T_C'].mean()
seq_df_mean.to_csv('data/tara/temperature/mae.csv')
print(mean_absolute_error(seq_df_mean.prediction,seq_df_mean.T_C))

 
