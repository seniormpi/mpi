import os
from traceback import print_tb
import binvox_rw
import numpy as np
import keras
import tensorflow
from keras.layers import MaxPooling3D, Dense, Flatten
from keras.layers.convolutional import Conv3D
from keras import regularizers
import pickle
import pandas as pd
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

model = keras.models.Sequential()
model.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape=(64, 64, 64, 1)))
model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model.add(Dense(2, activation='sigmoid'))
model.load_weights('model_mpi.h5')

model2 = keras.models.Sequential()
model2.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape=(64, 64, 64, 1)))
model2.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
model2.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model2.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model2.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model2.add(Flatten())
model2.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model2.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model2.add(Dense(1, activation='sigmoid'))
model2.load_weights('model_classify_mach_non_mach.h5')

isMach =  []
howMach = []
y_true = []
y_true2 = [1, #1
           1, #10
           1, #11
           1, #12
           1, #13
           1, #14
           1, #15
           1, #17
           2, #18
           2, #19
           1, #2
           2, #20
           2, #21
           2, #23
           3, #25
           3, #26
           3, #27
           3, #28
           3, #29
           1, #3
           2, #30
           2, #31
           3, #32
           2, #33
           2, #34
           3, #35
           3, #36
           3, #37
           2, #38
           1, #39
           1, #4
           3, #40
           2, #41
           3, #42
           1, #43
           1, #44
           2, #45
           2, #46
           1, #49
           1, #5
           1, #50
           3, #51
           2, #52
           2, #53
           1, #6
           1, #7
           1, #8
           1] #9
           

for filename in os.listdir('deneme'):
        batch_input = []
        with open('./deneme/' + filename, 'rb') as file:
            try:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
            except:
                print("-----------------------------" , filename)
                continue
        model_input = np.reshape(data, (64, 64, 64, 1))
        batch_input = [model_input]
        batch_x = np.array(batch_input)
        y_true.append(1)
        
        result2 = model2.predict_generator(batch_x)
        res2 = ""
        if (result2[:][result2[:] < 0.5]):
            res2 = "not machinable"
            isMach.append(0)
        if (result2[:][result2[:] >= 0.5]):
            res2 = "machinable"
            isMach.append(1)
          
        batch_input2 = []
        with open('./deneme/' + filename, 'rb') as file:
            try:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
            except:
                print("-----------------------------" , filename)
        model_input = np.reshape(data, (64, 64, 64, 1))
        batch_input2 = [model_input]
        batch_x = np.array(batch_input2)
        result = model.predict_generator(batch_x)
        res = ""
        count = 0
        if (result[:, 0][result[:, 0] >= 0.5]):  # = 1
            res += "Milling "
            count = 1
        if (result[:, 1][result[:, 1] >= 0.5]):  # = 1
            res += "Turning "
            if(count == 1):
                count = 3
            else:
                count = 2   
        if (result[:, 1][result[:, 1] < 0.5]):  # = 0
            res += " "
        if (result[:, 0][result[:, 0] < 0.5]):  # = 0
            res += " "
        howMach.append(count)
        print(filename, " " , res2 , "procidure: ", res)

y_true[40] = 0
y_true[35] = 0


# Define by steps in predict_generator * batch_size

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(isMach, y_true)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
lbl = ["Non-Machinable", "Machinable"]
df_cm = pd.DataFrame(cm, index=[i for i in lbl],columns=[i for i in lbl])
plt.figure(figsize=(10, 7))
cn_heat = sns.heatmap(df_cm, annot=True, cmap='Blues')
fig = cn_heat.get_figure()
fig.savefig('Mach_nonMach.png')


cm = confusion_matrix(howMach, y_true2)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
lbl = ["Milling", "Turning", "Milling-Turning"]
df_cm = pd.DataFrame(cm, index=[i for i in lbl],columns=[i for i in lbl])
plt.figure(figsize=(10, 7))
cn_heat = sns.heatmap(df_cm, annot=True, cmap='Blues')
fig = cn_heat.get_figure()
fig.savefig('Mach.png')