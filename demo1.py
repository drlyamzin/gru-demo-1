
# coding: utf-8

"""
Prediction of saccade times from an eye-tracking video. Input-output pairs are video-saccade labels.
"""

import os
os.getcwd()

from utils import *
from utils_gru import *
from load_and_preprocess import *
from callbacks import *
from keras.optimizers import SGD
import matplotlib.pyplot as plt


# load data (name of data file is hard-coded)
tensor, tensor_test, saccades, saccades_test, gcamp, gcamp_test = load_data()

# training set
input_data, target_data, gcamp_data = prep_train_test_data(tensor, saccades, gcamp, ndata=1000)

# test set
input_test, target_test, gcamp_test = prep_train_test_data(tensor_test, saccades_test, gcamp_test, ndata=100)

# train the model
# todo pack the below few lines into a separate fitting function
[_,h,w,T,_]=input_data.shape

sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9)

history = LossHistory()
savewgts = SaveWeights()
gru_model_cmp = model_gru(input_shape = (h, w, T, 1))
gru_model_cmp.compile(optimizer=sgd,
              loss='binary_crossentropy')
gru_model_cmp.fit(input_data, target_data, validation_data=(input_test,target_test), 
                  batch_size=20, epochs=100, verbose=2, callbacks = [history,savewgts])


# todo pack the below in a separate visualization/prediction function
input("Close figure and press Enter to continue...")

plt.figure(1)

plt.plot(history.val_losses)
plt.plot(history.losses)
plt.show()

input("Press Enter to continue...")

# test set: show predicted saccade time against ground truth

prd = gru_model_cmp.predict(input_test, verbose=2)
plot_trial = np.random.randint(100,size=(3,1))

plt.figure(2)

plt.subplot(131)
plt.plot(np.squeeze(prd[plot_trial[0],0,0,:,0]))
plt.plot(np.squeeze(target_test[plot_trial[0],0,0,:,0]))
plt.subplot(132)
plt.plot(np.squeeze(prd[plot_trial[1],0,0,:,0]))
plt.plot(np.squeeze(target_test[plot_trial[1],0,0,:,0]))
plt.subplot(133)
plt.plot(np.squeeze(prd[plot_trial[2],0,0,:,0]))
plt.plot(np.squeeze(target_test[plot_trial[2],0,0,:,0]))

plt.show()

input("Close figure and press Enter to continue...")

print("Model is being saved as 'gru_fitted'")


gru_model_cmp.save('gru_fitted')#redundant: history is being saved

# test set: show a video with predicted and ground truth saccade

yn = input("Make a video with a marked saccade prediction and ground truth? (y/n) \n")

if yn == 'y':
    im = plot_trial[1]
    ani=ani_frame_traintest(np.squeeze(input_test[im,:,:,:]),name='testvideo',sactime_true=np.squeeze(target_test[im,:,:,:]),
                        sactime_test=np.squeeze(prd[im,:,:,:]))

# eof