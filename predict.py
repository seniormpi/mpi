import binvox_rw
import numpy as np
import keras
from keras.layers import MaxPooling3D, Dense, Flatten
from keras.layers.convolutional import Conv3D
from keras import regularizers

class predict():
    
    model = keras.models.Sequential()
    model.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape = (64,64,64,1)))
    model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
    model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
    model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.0)))
    model.add(Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.0)))
    model.add(Dense(2, activation = 'sigmoid'))
    model.load_weights('models/model_mpi.h5')
    
    model2 = keras.models.Sequential()
    model2.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape = (64,64,64,1)))
    model2.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
    model2.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
    model2.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
    model2.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model2.add(Flatten())
    model2.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.0)))
    model2.add(Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.0)))
    model2.add(Dense(1, activation = 'sigmoid'))
    model2.load_weights('models/model_classify_mach_non_mach.h5')


    def predict_mpi(self):
        #predict of real parts with model 2        
        batch_input = []
        with open("./out/input.binvox", 'rb') as file:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
        model_input = np.reshape(data, (64, 64, 64, 1))
        batch_input += [model_input]
        batch_x = np.array(batch_input)
        
        result = self.model.predict(batch_x)
        res = ""
        if (result[:,0][result[:,0] >= 0.5]): #= 1
            res += "Milling "
        if (result[:,1][result[:,1] >= 0.5]): # = 1
            res += "Turning "
        if (result[:,1][result[:,1] < 0.5]): # = 0
            res += " "
        if (result[:,0][result[:,0] < 0.5]): # = 0
            res += " "
        print("Classification Results", res)
        
        return res
        
        
    def predict_mach(self):
        #predict of real parts with model 2        
        batch_input = []
        with open("./out/input.binvox", 'rb') as file:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
        model_input = np.reshape(data, (64, 64, 64, 1))
        batch_input += [model_input]
        batch_x = np.array(batch_input)
        
        result = self.model2.predict(batch_x)
        if (result[:][result[:] <0.5]):
            return "not machinable"
        if (result[:][result[:] >=0.5]):
            return "machinable"
        