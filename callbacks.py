
# coding: utf-8

# In[ ]:

# keras callbacks
import keras.callbacks as kc
import numpy as np

# In[ ]:

class LossHistory(kc.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        return
    
class SaveWeights(kc.Callback):
    def on_epoch_end(self, epoch, logs={}):
        fp='models_by_epoch/model{}'.format(epoch)
        self.model.save_weights(fp)

class SavePredictions(kc.Callback):
    def __init__(self, input_data):
        super().__init__()
        self.predhist = []
        self.input_data = input_data

    def on_epoch_end(self, epoch, logs={}):
        fp='predictions_by_epoch/model{}'.format(epoch)
        self.predhist.append(self.model.predict(self.input_data))
        print('length of predictions: {}'.format(len(self.predhist)))