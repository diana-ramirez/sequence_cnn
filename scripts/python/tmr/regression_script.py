# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 07:54:42 2020

@author: Peng
"""
#%%
# Libraries and modules
# import pandas as pd
import numpy as np

import sys

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf

#%%
#Inputs
data_path='data/cluster_dataframes/'
regress_var='T_C'
model_save_path='data/models/tara_T_regression'

#nn parameters
max_len=1500
sample_n=10
embed_size = 256
batch_size=100
epochs=100
cnn_fun_path='scripts/python/tmr/'
seq_type='aa'
num_letters=26
seq_resize=True 

#%%
#generate datasets for fitting
seq_df=cf.load_seq_dataframe(data_path)

#%%
#generate training data for annotation/cluster datasets
##annotations
train=seq_df.groupby(['Site']).sample(n=sample_n)
seq_df=seq_df.drop(train.index)
train_one_hot=cf.seq_one_hot(seq_df['sequence'],
                              seq_type=seq_type,
                              max_len=max_len,
                              seq_resize=seq_resize)
ytrain=np.array(train[regress_var],dtype=float)

#%%
#generate validation data for annotation/cluster datasets
##annotation
validation=seq_df.groupby(['Site']).sample(n=sample_n)
seq_df=seq_df.drop(validation.index)
validation_one_hot=cf.seq_one_hot(validation['sequence'],
                              seq_type=seq_type,
                              max_len=max_len,
                              seq_resize=seq_resize)
yvalidation=np.array(validation[regress_var],dtype=float)

#generate test data for annotation/cluster datasets
##annotation

test=seq_df
test_one_hot=cf.seq_one_hot(test['sequence'],
                              seq_type=seq_type,
                              max_len=max_len,
                              seq_resize=seq_resize)
ytest=np.array(test[regress_var],dtype=float)

del seq_df
  
# sequence_length = train_one_hot.shape[1]
model= cf.regression_blstm(num_letters,
                           max_len,
                           embed_size=embed_size)

n_validation_a=validation_one_hot.shape[0]

model.fit(x=train_one_hot,y=ytrain,
          batch_size=batch_size,
          validation_data=(validation_one_hot,yvalidation),
          epochs=epochs)

model.save(model_save_path + '.h5')

model.evaluate(test_one_hot,ytest)





