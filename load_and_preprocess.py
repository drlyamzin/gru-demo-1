
# coding: utf-8


import scipy.io as sio
import numpy as np
from utils import datajitter_wgcamp


def load_data():    
    # load data from .mat files: preprocessed (cropped) eye videos, saccade labels, gcamp signal from the brain (not used in this demo)
    mat_contents = sio.loadmat('data\\frames_200_trials.mat')
    tensor=mat_contents['frames']
    mat_contents = sio.loadmat('data\\saccades_and_gcamp.mat')
    saccades = mat_contents['sac_tr']
    gcamp = mat_contents['tensor_tr']
    #eyemv = mat_contents['eyemv_tr']

    # loaded video data must be already preprocessed and cropped to sizes (n,H,W,T)=(200,39,39,45)
    # size of saccades (n,T)=(1,45)
    # size of gcamp (n,K,T)=(200,1,45) with K reserved for K-dimensional output
    
    # train/test sets:
    tensor_test = tensor[151:,:,:,:]
    saccades_test = saccades[151:,:]
    gcamp_test = gcamp[151:,:,:]
    #eyemv_test = eyemv[151:,:]

    tensor = tensor[:151,:,:,:]
    saccades = saccades[:151,:]
    gcamp = gcamp[:151,:,:]
    #eyemv = eyemv[:151,:]
    
    return (tensor, tensor_test, saccades, saccades_test, gcamp, gcamp_test)#, eyemv, eyemv_test)

def load_data_dict():    
    # load data from .mat files: preprocessed (cropped) eye videos, saccade labels, gcamp signal from the brain (not used in this demo)
    mat_contents = sio.loadmat('data\\frames_200_trials.mat')
    tensor=mat_contents['frames']
    mat_contents = sio.loadmat('data\\saccades_and_gcamp.mat')
    saccades = mat_contents['sac_tr']
    gcamp = mat_contents['tensor_tr']
    #eyemv = mat_contents['eyemv_tr']

    # loaded video data must be already preprocessed and cropped to sizes (n,H,W,T)=(200,39,39,45)
    # size of saccades (n,T)=(1,45)
    # size of gcamp (n,K,T)=(200,1,45) with K reserved for K-dimensional output
    
    data_dict = {}
    # train/test sets:
    data_dict['tensor_test'] = tensor[151:,:,:,:]
    data_dict['saccades_test'] = saccades[151:,:]
    data_dict['gcamp_test'] = gcamp[151:,:,:]
    #eyemv_test = eyemv[151:,:]

    data_dict['tensor'] = tensor[:151,:,:,:]
    data_dict['saccades'] = saccades[:151,:]
    data_dict['gcamp'] = gcamp[:151,:,:]
    #eyemv = eyemv[:151,:]
    
    return data_dict

def prep_train_test_data(tensor, saccades, gcamp, ndata=1000): #eyemv,
    
    # prepare batches by bootstrap and jitter
    # adjust dimensions to prepare for use by model
    
    N = tensor.shape[0]
    h = tensor.shape[1]
    w = tensor.shape[2]
    T = tensor.shape[3]
    #ndata = 1000# examples

    input_data = [None]*ndata
    target_data = [None]*ndata
    gcamp_data = [None]*ndata
    #eyemv_data = [None]*ndata

    trial_ids = np.random.randint(0,N,size=(ndata,))
    t_in=np.squeeze(tensor[trial_ids,:,:,:])
    s_in=np.squeeze(saccades[trial_ids,:])
    g_in=gcamp[trial_ids,:,:]
    #e_in=eyemv[trial_ids,:]

    input_data, target_data, gcamp_data = datajitter_wgcamp(t_in,s_in,g_in)

    h = input_data.shape[1]
    w = input_data.shape[2]
    T = input_data.shape[3]

    # expand dimensions of targets, and once - of input
    input_data = np.expand_dims(input_data,4)
    target_data = np.expand_dims(target_data,1)
    target_data = np.expand_dims(target_data,2)
        # if scrambled time:
        #target_batches[i] = np.expand_dims(target_batches[i],3)# time in the input is (3) in the output (4)
        # if preserved time:
    target_data = np.expand_dims(target_data,4)
    
    gcamp_data = np.squeeze(gcamp_data,axis=1)
    gcamp_data = np.expand_dims(gcamp_data,2)
    
    return input_data, target_data, gcamp_data

def prep_train_test_data_dict(tensor, saccades, gcamp, ndata=1000): #eyemv,
    
    # prepare batches by bootstrap and jitter
    # adjust dimensions to prepare for use by model
    
    N = tensor.shape[0]
    h = tensor.shape[1]
    w = tensor.shape[2]
    T = tensor.shape[3]
    #ndata = 1000# examples

    input_data = [None]*ndata
    target_data = [None]*ndata
    gcamp_data = [None]*ndata
    #eyemv_data = [None]*ndata

    trial_ids = np.random.randint(0,N,size=(ndata,))
    t_in=np.squeeze(tensor[trial_ids,:,:,:])
    s_in=np.squeeze(saccades[trial_ids,:])
    g_in=gcamp[trial_ids,:,:]
    #e_in=eyemv[trial_ids,:]

    input_data, target_data, gcamp_data = datajitter_wgcamp(t_in,s_in,g_in)

    h = input_data.shape[1]
    w = input_data.shape[2]
    T = input_data.shape[3]

    # expand dimensions of targets, and once - of input
    input_data = np.expand_dims(input_data,4)
    target_data = np.expand_dims(target_data,1)
    target_data = np.expand_dims(target_data,2)
        # if scrambled time:
        #target_batches[i] = np.expand_dims(target_batches[i],3)# time in the input is (3) in the output (4)
        # if preserved time:
    target_data = np.expand_dims(target_data,4)
    
    gcamp_data = np.squeeze(gcamp_data,axis=1)
    gcamp_data = np.expand_dims(gcamp_data,2)
    
    prep_data_dict={}
    prep_data_dict['input_data'] = input_data
    prep_data_dict['target_data'] = target_data
    prep_data_dict['gcamp_data'] = gcamp_data
    
    return prep_data_dict