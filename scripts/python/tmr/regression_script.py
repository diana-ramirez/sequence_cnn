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
data_path='data/tara/'
regress_var='T_C'
model_save_path='data/models/tara_T_regression'

#nn parameters
max_len=500
sample_f=0.1
embed_size = 256
batch_size=100
epochs=100
cnn_fun_path='scripts/python/tmr/'
seq_type='aa'
num_letters=26
seq_resize=False

#%%
#generate datasets for fitting
seq_df=cf.load_seq_dataframe(data_path)
seq_df=seq_df.sample(frac=0.1)

#%%
#generate training data for annotation/cluster datasets
##annotations
train=seq_df.groupby(['Site']).sample(frac=sample_f)
seq_df=seq_df.drop(train.index)

print('Starting to one-hot encode sequences...')

train_one_hot=cf.seq_one_hot(train['sequence'],
                              seq_type=seq_type,
                              max_len=max_len,
                              seq_resize=seq_resize)
ytrain=np.asarray(train[regress_var],dtype=float)
#%%
#generate validation data for annotation/cluster datasets
##annotation
validation=seq_df.groupby(['Site']).sample(frac=sample_f)
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

print("Starting to fit model...")

model.fit(x=train_one_hot,y=ytrain,
          batch_size=batch_size,
          validation_data=(validation_one_hot,yvalidation),
          epochs=epochs)

model.save(model_save_path + '.h5')

model.evaluate(test_one_hot,ytest)





