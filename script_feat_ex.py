import numpy as np
import tensorflow as tf
import keras
from keras import regularizers
import os
import random
from keras.layers import Input, MaxPooling3D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import Conv3D
from keras.models import Model
from keras import optimizers
from keras import regularizers
import pandas as pd
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
from livelossplot import PlotLossesKeras
import tikzplotlib
import binvox_rw
import os

import warnings
warnings.filterwarnings("ignore")

# Part - 1 : Feature Classification
def make_labels():
    cols = ['model', 'Feature']
    lst = []
    for feat_num in range(1, 22):
        if feat_num == 1:
            print("skipped")
            continue
        else:
            for model_num in range(1, 201):
                for orn in range(1, 7):
                    lst.append([str(feat_num) + "_" + str(model_num) + "_" + str(orn), feat_num])
    df1 = pd.DataFrame(lst, columns=cols)
    df = df1
    return df

# data process
def data_process(dataframe, num_features=20):
    # process the batch data
    np.random.seed(123)
    validate = []
    train = []
    test = []
    print(dataframe)
    for f in range(num_features):
        array1 = dataframe['model'].tolist()[(f * 1200):((f + 1) * 1200)]
        array2 = [s + '.binvox' for s in array1]
        random.shuffle(array2)
        train = train + array2[0:840]
        validate = validate + array2[840:1020]
        test = test + array2[1020:1200]

    partition = {'train': train, 'validate': validate, 'test': test}
    return partition

# Model Generator
def model_generator(data_dir, index_list, dataframe, mode, batch_size=32):
    np.random.seed(123)
    one_hot = np.identity(21)
    global Y_true
    counter = 0
    while True:
        batch_paths = np.random.choice(a=index_list, size=batch_size)
        batch_input = []
        batch_output = []
        for input_path in batch_paths:
            file_path = os.path.join(data_dir, input_path)
            with open(file_path, 'rb') as file:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
            model_input = np.reshape(data, (64, 64, 64, 1))
            index_num = int(input_path.split('_')[0])
            label = one_hot[index_num - 1, :]
            model_output = label.tolist()

            batch_input += [model_input]
            batch_output += [model_output]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        if (mode == 'test'):
            Y_true += batch_output
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield (batch_x, batch_y)
        

data_dir = '.\\data\\feature_data'

df = make_labels()
partition = data_process(df)
Y_true = []


training_generator = model_generator(data_dir, partition['train'], df, 'train')
validation_generator = model_generator(data_dir, partition['validate'], df, 'validate')
test_generator = model_generator(data_dir, partition['test'], df, 'test')

from livelossplot import PlotLossesKeras

model = tf.keras.models.Sequential()
model.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape=(64, 64, 64, 1)))
model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model.add(Dense(21, activation='softmax'))

learning_rate = 0.0005
decay_rate = 0
adam = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
              metrics=['accuracy'])

model.summary()

history1 = model.fit_generator(training_generator,
                               steps_per_epoch=5,
                               validation_data=validation_generator,
                               validation_steps=5,
                               epochs=10)

# Saving all Models and Accuracies
model.save_weights('model_features.h5')

# Confusion Matrix
model.load_weights('model_features.h5')
y_pred = model.predict_generator(test_generator, steps=10)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.array(Y_true)
y_true = np.argmax(y_true, axis=1)
# Define by steps in predict_generator * batch_size
y_true_cut = y_true[0:10 * 32]
print(y_true_cut.shape)
print('---------------------')
print(y_pred.shape)

print(np.sum(y_pred == y_true_cut))
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true_cut, y_pred)
np.savetxt('part1_cm_feature.txt', cm)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
lbl = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
       "21"]
df_cm = pd.DataFrame(cm, index=[i for i in lbl],
                     columns=[i for i in lbl])
plt.figure(figsize=(10, 7))
cn_heat = sns.heatmap(df_cm, annot=True, cmap='Blues')
fig = cn_heat.get_figure()
fig.savefig('part1_conf_mat_features.png')

# Part 2 : Machining Process Recognition

def make_labels():
    cols = ['model', 'Milling', 'Turning']
    lst = []
    for feat_num in range(1, 4):
        for model_num in range(1, 1001):
            for orn in range(1, 7):
                if feat_num == 1:
                    lst.append([str(feat_num) + "_" + str(model_num) + "_" + str(orn), 1, 0])
                if feat_num == 2:
                    lst.append([str(feat_num) + "_" + str(model_num) + "_" + str(orn), 0, 1])
                if feat_num == 3:
                    lst.append([str(feat_num) + "_" + str(model_num) + "_" + str(orn), 1, 1])

    df1 = pd.DataFrame(lst, columns=cols)
    df = df1
    return df

def data_process(dataframe):
    # process the batch data
    validate = []
    train = []
    test = []
    for f in range(3):
        array1 = dataframe['model'].tolist()[(f * 6000):((f + 1) * 6000)]
        array2 = [s + '.binvox' for s in array1]
        random.shuffle(array2)
        train = train + array2[0:4200]
        validate = validate + array2[4200:5100]
        test = test + array2[5100:6000]

    partition = {'train': train, 'validate': validate, 'test': test}
    return partition

def model_generator(data_dir, index_list, dataframe, mode, batch_size=32):
    global Y_true
    while True:
        batch_paths = np.random.choice(a=index_list, size=batch_size)
        batch_input = []
        batch_output = []

        for input_path in batch_paths:
            file_path = os.path.join(data_dir, input_path)
            with open(file_path, 'rb') as file:
                try:
                    data = np.int32(binvox_rw.read_as_3d_array(file).data)
                except:
                    print(file)
            model_input = np.reshape(data, (64, 64, 64, 1))
            data2 = input_path.replace('.binvox', '')
            label = dataframe.loc[dataframe['model'] == data2, ['Milling', 'Turning']]
            model_output = label.values.tolist()[0]

            batch_input += [model_input]
            batch_output += [model_output]

        if (mode == 'test'):
            Y_true += batch_output
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)

data_dir = '.\\data\\cmb_data'
df = make_labels()
partition = data_process(df)

Y_true = []
training_generator = model_generator(data_dir, partition['train'], df, 'train')
validation_generator = model_generator(data_dir, partition['validate'], df, 'validate')
test_generator = model_generator(data_dir, partition['test'], df, 'test')

model = keras.models.Sequential()
model.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape=(64, 64, 64, 1)))
model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(21, activation='softmax'))

for layer in model.layers[:4]:
    layer.trainable = False

model.load_weights('model_features.h5')
model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
model.pop()

print(model.summary())
model1 = model
model1.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model1.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model1.add(Flatten())
model1.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model1.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model1.add(Dense(2, activation='sigmoid'))

learning_rate = 0.0001
decay_rate = 0
adam = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)
model1.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
               metrics=['accuracy'])

model1.summary()

history = model1.fit_generator(generator=training_generator,
                               steps_per_epoch=5,
                               validation_data=validation_generator,
                               validation_steps=5,
                               epochs=500)

model1.save_weights("model_mpi.h5")
score, acc = model1.evaluate_generator(test_generator, steps=10)
print(acc)

# Part 3 : Machinability classification of Synthetic Parts

def make_labels():
    cols = ['model', 'machinable']
    lst = []
    for feat_num in range(1, 4):
        for model_num in range(1, 1001):
            for orn in range(1, 7):
                lst.append([str(feat_num) + "_" + str(model_num) + "_" + str(orn), 1])

    for feat_num in range(4, 6):
        for model_num in range(1, 801):
            for orn in range(1, 7):
                lst.append([str(feat_num) + "_" + str(model_num), 0])

    for feat_num in range(6, 7):
        for model_num in range(1, 801):
            for orn in range(1, 7):
                lst.append([str(feat_num) + "_" + str(model_num) + "_" + str(orn), 0])

    df1 = pd.DataFrame(lst, columns=cols)
    df = df1
    return df

def data_process(dataframe):
    # process the batch data
    validate = []
    train = []
    test = []
    for f in range(1, 4):
        array1 = dataframe['model'].tolist()[(f * 6000):((f + 1) * 6000)]
        array2 = [s + '.binvox' for s in array1]
        random.shuffle(array2)
        train = train + array2[0:4200]
        validate = validate + array2[4200:5100]
        test = test + array2[5100:6000]

    for f in range(4, 6):
        array1 = dataframe['model'].tolist()[(f * 4800):((f + 1) * 4800)]
        array2 = [s + '.binvox' for s in array1]
        random.shuffle(array2)
        train = train + array2[0:3360]
        validate = validate + array2[3360:4080]
        test = test + array2[4080:4800]

    for f in range(6, 7):
        array1 = dataframe['model'].tolist()[(f * 4800):((f + 1) * 4800)]
        array2 = [s + '.binvox' for s in array1]
        random.shuffle(array2)
        train = train + array2[0:3360]
        validate = validate + array2[3360:4080]
        test = test + array2[4080:4800]

    partition = {'train': train, 'validate': validate, 'test': test}
    return partition

def model_generator(data_dir, index_list, dataframe, mode, batch_size=32):
    global Y_true
    while True:
        batch_paths = np.random.choice(a=index_list, size=batch_size)
        batch_input = []
        batch_output = []

        for input_path in batch_paths:
            file_path = os.path.join(data_dir, input_path)
            with open(file_path, 'rb') as file:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
            model_input = np.reshape(data, (64, 64, 64, 1))
            data2 = input_path.replace('.binvox', '')
            label = dataframe.loc[dataframe['model'] == data2]['machinable']
            # print(label)
            model_output = label.values.tolist()[0]
            # print(model_output)

            batch_input += [model_input]
            batch_output += [model_output]

        if (mode == 'test'):
            Y_true += batch_output
        #             print("I got index", index, " and ", len(index_list))
        #             index += 1
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)

data_dir = '.\\data\\cmb_data_machinability'
df = make_labels()
partition = data_process(df)

Y_true = []

training_generator = model_generator(data_dir, partition['train'], df, 'train')
validation_generator = model_generator(data_dir, partition['validate'], df, 'validate')
test_generator = model_generator(data_dir, partition['test'], df, 'test')

model = keras.models.Sequential()
model.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape=(64, 64, 64, 1)))
model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(21, activation='softmax'))

for layer in model.layers[:4]:
    layer.trainable = False

model.load_weights('model_features.h5')
model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
print(model.summary())

model1 = model

model1.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
model1.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model1.add(Flatten())
model1.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model1.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model1.add(Dense(1, activation='sigmoid'))

learning_rate = 0.0001
decay_rate = 0
adam = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)
model1.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
               metrics=['accuracy'])

#history = model1.fit(x_train, y_train, validation_data=(x_val, y_val), steps_per_epoch=5, validation_steps=5, epochs=500)

history = model1.fit_generator(training_generator,
                     steps_per_epoch=5,
                     validation_data=validation_generator,
                     validation_steps=5,
                     epochs=500)

model1.save_weights('model_classify_mach_non_mach.h5')