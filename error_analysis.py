
# coding: utf-8

# In[ ]:


from utils import *
from utils_gru import *
from load_and_preprocess import *
import time
import matplotlib.pyplot as plt

# complete code to report the mse on N iterations of the model

def errorByEpoch(input_data,target_data,input_test,target_test,folder_id="models_by_epoch"):
# calculate training and testing errors

    out_weights = []
    out_test_mse = []
    out_train_mse = []
    out_test_xe = []
    out_train_xe = []
    
    out_train_xe_trials = []
    out_train_se_trials = []
    out_test_xe_trials = []
    out_test_se_trials = []
    
    out_test_fneg = []
    out_train_fneg = []
    out_test_fpos = []
    out_train_fpos = []
    out_test_tpos = []
    out_train_tpos = []
    out_test_tneg = []
    out_train_tneg = []
    
    
    model_tmp = model_gru((32,32,39,1))# note that for GCaMP version we have (32,32,None,1) - adjust model_gru code if flexible time is necessary
    
    for imodel in range(10):
        # load the model and get the weights
        start = time.time()
        model_tmp.load_weights('{}/model{}'.format(folder_id,imodel))
        end = time.time()
        print("Load weights: Model {0}: seconds {1}".format(imodel, end - start))

        start = time.time()
        weights = model_tmp.get_weights()
        end = time.time()

        print("Get weights from Model {0}: seconds {1}".format(imodel, end - start))
        
        # predict train and test targets
        start = time.time()
        pred_test = model_tmp.predict(input_test,batch_size=20)
        pred_train = model_tmp.predict(input_data,batch_size=20)
        end = time.time()
        
        # calculate basic error metrics: false negatives, false positives; and correct outcomes: true detections, true rejections 
        fneg_test = falseneg(pred_test,target_test)
        fneg_train = falseneg(pred_train,target_data)
        fpos_test = falsepos(pred_test,target_test)
        fpos_train = falsepos(pred_train,target_data)
        tpos_test = truepos(pred_test,target_test)
        tpos_train = truepos(pred_train,target_data)
        tneg_test = trueneg(pred_test,target_test)
        tneg_train = trueneg(pred_train,target_data)
        
        print("test set false negatives {}".format(fneg_test))
        print("train set false negatives {}".format(fneg_train))

        print("Predict train and test from Model {0}: seconds {1}".format(imodel, end - start))

        # calculate training and testing sets mse, on all trials/examples
        start = time.time()
        test_mse = np.ndarray.flatten(pred_test)-np.ndarray.flatten(target_test)
        test_mse = np.nanmean(np.square(test_mse))
        end = time.time()

        print("Calculate test MSE, seconds {0}".format(end - start))

        start = time.time()
        train_mse=np.ndarray.flatten(pred_train)-np.ndarray.flatten(target_data)
        train_mse = np.nanmean(np.square(train_mse))
        end = time.time()

        print("Calculate test MSE, seconds {0}".format(end - start))

        # calculate training and testing sets cross-entropy, on all trials/examples
        start = time.time()
        t = np.ndarray.flatten(target_test)
        p = np.ndarray.flatten(pred_test)
        test_xe = (np.dot(t,np.log(p)) + np.dot(1-t,np.log(1-p)) )/input_test.shape[0]
        end = time.time()
        
        print("Calculate test X-entropy, seconds {0}".format(end - start))
        
        start = time.time()
        t = np.ndarray.flatten(target_data)
        p = np.ndarray.flatten(pred_train)
        train_xe = (np.dot(t,np.log(p)) + np.dot(1-t,np.log(1-p)) )/input_data.shape[0]
        end = time.time()
        
        print("Calculate train X-entropy, seconds {0}".format(end - start))
        
        # calculate training and testing sets se and xe on individual trials 
        t = np.ndarray.reshape(target_data,target_data.shape[0],-1)
        p = np.ndarray.reshape(pred_train,pred_train.shape[0],-1)
        train_se_trials = np.einsum('ij,ij->i',t-p,t-p)
        train_xe_trials = (np.einsum('ij,ij->i',t,np.log(p)) + np.einsum('ij,ij->i',1-t,np.log(1-p)) )
        
        t = np.ndarray.reshape(target_test,target_test.shape[0],-1)
        p = np.ndarray.reshape(pred_test,pred_test.shape[0],-1)
        test_se_trials = np.einsum('ij,ij->i',t-p,t-p)
        test_xe_trials = (np.einsum('ij,ij->i',t,np.log(p)) + np.einsum('ij,ij->i',1-t,np.log(1-p)) )
        
        out_test_fneg.append(fneg_test)
        out_train_fneg.append(fneg_train)
        out_test_fpos.append(fpos_test)
        out_train_fpos.append(fpos_train)
        out_test_tpos.append(tpos_test)
        out_train_tpos.append(tpos_train)
        out_test_tneg.append(tneg_test)
        out_train_tneg.append(tneg_train)
        
        out_train_mse.append(train_mse)
        out_test_mse.append(test_mse)
        out_train_xe.append(train_xe)
        out_test_xe.append(test_xe)
        
        out_train_xe_trials.append(train_xe_trials)
        out_train_se_trials.append(train_se_trials)
        out_test_xe_trials.append(test_xe_trials)
        out_test_se_trials.append(test_se_trials)
        
        out_weights.append(np.squeeze(weights[-1]))
        
    #plt.plot(out_test_mse)
    #plt.plot(out_train_mse)

    return out_train_mse, out_test_mse, out_train_xe, out_test_xe, out_train_xe_trials, out_train_se_trials, out_test_xe_trials, out_test_se_trials, out_test_fneg, out_train_fneg, out_test_fpos, out_train_fpos, out_test_tpos, out_train_tpos, out_test_tneg, out_train_tneg, out_weights

def falseneg(prediction,ground_truth,threshold=0.5):
    # simple false negative, all time bins are treated equally as data points
    
    # threshold at 50% by default
    ind = ground_truth==1.0
    fneg = np.sum(prediction[ind]<threshold)/np.sum(ind)
    
    return fneg

def falsepos(prediction,ground_truth):
    # threshold at 50%
    ind = ground_truth==0.0
    fpos = np.sum(prediction[ind]>=0.5)/np.sum(ind)
    
    return fpos

def truepos(prediction,ground_truth):
    # threshold at 50%
    ind = ground_truth==1.0
    tpos = np.sum(prediction[ind]>=0.5)/np.sum(ind)
    
    return tpos

def trueneg(prediction,ground_truth):
    # threshold at 50%
    ind = ground_truth==0.0
    tneg = np.sum(prediction[ind]<0.5)/np.sum(ind)
    
    return tneg

# TP,FP,TN,FN computed based on the prediction being at least once within the saccade period or outside of it:

def fn_by_period(prediction,ground_truth,threshold=0.5):
    
    ind = ground_truth==1.0
    # if the prediction never exceeds threshold within the saccade period, consider it a FN
    fneg = (np.sum(prediction[ind]>=threshold))==0
    
    return fneg

def fp_by_period(prediction,ground_truth,threshold=0.5):
    
    ind = ground_truth==0.0
    # if the prediction exceeds threshold at least once within the no-saccade period, consider it a FP
    fpos = (np.sum(prediction[ind]>=threshold))>0
    
    return fpos

def tp_by_period(prediction,ground_truth,threshold=0.5):
    
    ind = ground_truth==1.0
    # if the prediction exceeds threshold at least once within the saccade period, consider it a TP
    tpos = (np.sum(prediction[ind]>=threshold))>0
    
    return tpos


def tn_by_period(prediction,ground_truth,threshold=0.5):
    
    ind = ground_truth==0.0
    # if the prediction exceeds threshold at least once within the saccade period, consider it a TP
    tneg = (np.sum(prediction[ind]>=threshold))==0
    
    return tneg

def plotall(out_train_mse,out_test_mse,out_train_xe,out_test_xe,out_train_se_trials,out_test_se_trials,out_train_xe_trials,
           out_test_xe_trials):
    fig1 = plt.figure(1)
    plt.plot(out_train_mse, label='train mse per timebin')
    plt.plot(out_test_mse, label='test mse per timebin')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    fig1.legend()
    
    fig2 = plt.figure(2)
    plt.plot(out_train_xe, label='train cross-ent per trial')
    plt.plot(out_test_xe, label='test cross-ent per trial')
    plt.xlabel('epoch')
    plt.ylabel('cross-ent')
    fig2.legend()
    
    fig3 = plt.figure(3)
    for i,val in enumerate(out_train_se_trials):
        h=plt.scatter(i*np.ones(100,)+0.1*np.random.randn(100,),val[::10],marker='x',color='b') 
        if i==0: 
            h.set_label('train se by trial')
    for i,val in enumerate(out_test_se_trials):
        h=plt.scatter(i*np.ones(100,)+0.1*np.random.randn(100,),val,marker='x',color='r')
        if i==0:
            h.set_label('test se by trial')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    fig3.legend()
    
    fig4 = plt.figure(4)
    for i,val in enumerate(out_train_xe_trials):
        h=plt.scatter(i*np.ones(100,)+0.1*np.random.randn(100,),val[::10],marker='x',color='b') 
        if i==0: 
            h.set_label('train cross-ent by trial')
    for i,val in enumerate(out_test_xe_trials):
        h=plt.scatter(i*np.ones(100,)+0.1*np.random.randn(100,),val,marker='x',color='r')
        if i==0: 
            h.set_label('test cross-ent by trial')
    plt.xlabel('epoch')
    plt.ylabel('cross-ent')
    fig4.legend()
    
    plt.show() 

    
if __name__ == "__main__":
    
    # load data (name of data file is hard-coded)
    tensor, tensor_test, saccades, saccades_test, gcamp, gcamp_test = load_data()

    # training set
    input_data, target_data, gcamp_data = prep_train_test_data(tensor, saccades, gcamp, ndata=1000)

    # test set
    input_test, target_test, gcamp_test = prep_train_test_data(tensor_test, saccades_test, gcamp_test, ndata=100)
    
    # calculate errors
    out_train_mse, out_test_mse, out_train_xe, out_test_xe, out_train_xe_trials, out_train_se_trials, out_test_xe_trials, out_test_se_trials, _, _, _, _, _, _, _, _, _ = errorByEpoch(input_data,target_data,input_test,target_test)
    
    # plot figures
    plotall(out_train_mse,out_test_mse,out_train_xe,out_test_xe,out_train_se_trials,out_test_se_trials,out_train_xe_trials,
           out_test_xe_trials)
