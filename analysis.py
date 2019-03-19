from load_and_preprocess import *
import matplotlib.pyplot as plt
from utils_gru import *
import random
import math
import argparse
from demo_main import *
from error_analysis import fn_by_period,fp_by_period,tn_by_period,tp_by_period

def confusion_matrix(prediction,data):

    fn = []
    tn = []
    fp = []
    tp = []
    for trial in range(prediction.shape[0]):
        fn.append(fn_by_period(np.squeeze(prediction[trial,:,:,:,:]),np.squeeze(data[trial,:,:,:,:])))
        fp.append(fp_by_period(np.squeeze(prediction[trial,:,:,:,:]),np.squeeze(data[trial,:,:,:,:])))
        tn.append(tn_by_period(np.squeeze(prediction[trial,:,:,:,:]),np.squeeze(data[trial,:,:,:,:])))
        tp.append(tp_by_period(np.squeeze(prediction[trial,:,:,:,:]),np.squeeze(data[trial,:,:,:,:])))
    
    tp_sum,fp_sum,tn_sum,fn_sum = map(sum,[tp,fp,tn,fn])
    
    return tp_sum,fp_sum,tn_sum,fn_sum

def confusion_by_epoch(target_test,input_test):

    # loop over epochs, predict, calculate tp and fp, accumulate
    tp=[]
    fp=[]
    tn=[]
    fn=[]

    ground_truth = target_test
        
    for epoch_id in range(0,1000,9):
        print(epoch_id)
        pred = predict_from_epoch(epoch_id, input_test)

        tp_this,fp_this,tn_this,fn_this = confusion_matrix(pred,ground_truth)
        
        tp.append(tp_this)
        fp.append(fp_this)
        tn.append(tn_this)
        fn.append(fn_this)

    return tp,fp,tn,fn
    

def main():

    input_data, target_data, input_test, target_test = load_default_data()

    tp,fp,tn,fn = confusion_by_epoch(target_test,input_test)

    plt.plot(tp)
    plt.plot(fp)

    plt.show()

    pass

if __name__=="__main__":

    main()

    pass