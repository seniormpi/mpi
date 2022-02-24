import os
from traceback import print_tb
import binvox_rw
import numpy as np
import keras
import tensorflow
from keras.layers import MaxPooling3D, Dense, Flatten
from keras.layers.convolutional import Conv3D
from keras import regularizers

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
model.load_weights('models/model_mpi.h5')

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
model2.load_weights('models/model_classify_mach_non_mach.h5')

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

        result2 = model2.predict(batch_x)
        res2 = ""
        if (result2[:][result2[:] < 0.5]):
            res2 = "not machinable"
        if (result2[:][result2[:] >= 0.5]):
            res2 = "machinable"

        if (res2 == "not machinable"):
            print(filename, " " , res2)
        else:    
            batch_input2 = []
            with open('./deneme/' + filename, 'rb') as file:
                try:
                    data = np.int32(binvox_rw.read_as_3d_array(file).data)
                except:
                    print("-----------------------------" , filename)
            model_input = np.reshape(data, (64, 64, 64, 1))
            batch_input2 = [model_input]
            batch_x = np.array(batch_input2)
            result = model.predict(batch_x)
            res = ""
            if (result[:, 0][result[:, 0] >= 0.5]):  # = 1
                res += "Milling "
            if (result[:, 1][result[:, 1] >= 0.5]):  # = 1
                res += "Turning "
            if (result[:, 1][result[:, 1] < 0.5]):  # = 0
                res += " "
            if (result[:, 0][result[:, 0] < 0.5]):  # = 0
                res += " "
            print(filename, " " , res2 , "procidure: ", res)