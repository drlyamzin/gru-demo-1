
# coding: utf-8

from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Masking, Conv1D, Conv3D, Reshape, Permute, MaxPooling3D, Lambda
from keras.layers import GRU, BatchNormalization, Reshape
from keras.initializers import glorot_uniform, glorot_normal
from keras.backend import int_shape
from keras.regularizers import l2, l1
import keras.backend as K
import tensorflow as tf


def model_gru(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    strd = (2,2,1)# strides for maxpooling
    sz = (3,3,3)# size of filter in stackblock
    
    X_input = Input(shape = input_shape)
    
    H,W,T,_ = input_shape
        
    X=stackBlock(X_input,4,sz,1)
    X=stackBlock(X,4,sz,2)
    X = MaxPooling3D((2, 2, 1), strides=strd)(X)
    
    X=stackBlock(X,4,sz,3)
    X=stackBlock(X,4,sz,4)
    X = MaxPooling3D((2, 2, 1), strides=strd)(X)
    
    X=stackBlock(X,4,sz,5)
    X=stackBlock(X,4,sz,6)
    X = MaxPooling3D((2, 2, 1), strides=strd)(X)
    
    sp = int_shape(X)
    print(sp)
    
    # (m),H,W,T,C: must transform into (batch_size, timesteps, input_dim)
    
    X = Permute((3,1,2,4))(X)
    X = Reshape((T,-1))(X)
    
    sp = int_shape(X)
    print(sp)
    
    # Step 3: First GRU Layer
    X = GRU(32, return_sequences=True)(X)         # GRU (use 32 units and return the sequences)
    X = Dropout(0.2)(X)                                
    X = BatchNormalization()(X)                                
    
    # Step 4: Second GRU Layer 
    X = GRU(32, return_sequences=True)(X)         # GRU (use 32 units and return the sequences)
    X = Dropout(0.2)(X)                                
    X = BatchNormalization()(X)                     
    X = Dropout(0.2)(X)                               
    
    # Step 5: Time-distributed dense layer 
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    sp = int_shape(X)
    print(sp)
    
    X = Reshape((T,1,1,1))(X)
    X = Permute((2,3,1,4))(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    return model  

def stackBlock(X, f, sz, stage, strd = (1,1,1)):
    """        
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, nT_prev, n_C_prev) 
    f -- n output filters
    stage -- number of stage, part of naming 
    
    Returns:
    X -- output of stackBlock, tensor of shape (m, n_H_prev, n_W_prev, nT_prev, f) 
    """
    
    X = Conv3D(f, sz, strides = strd, padding='same', name = 'conv' + str(stage), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # do BN on channels (be sure that the chosen axis is channels!)
    X = BatchNormalization(axis = 4, name = 'bn_' + str(stage))(X)
    X = Activation('relu')(X)
    
    # the pooling step is moved outside the block, so that several convolutions can be done before pooling
    return X