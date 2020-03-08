import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)
import os
from keras.preprocessing.image import ImageDataGenerator
import cv2
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, TimeDistributed, CuDNNLSTM, CuDNNGRU
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
import threading
import time
import shutil
import pickle
from utils.data_generator import DataGenerator
from utils.eta_tool import ETATool
from utils.basedir import BASEDIR


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
# set_session(tf.Session(config=config))

os.chdir(BASEDIR)

def crop_image(img_array):
    h_crop = 175  # horizontal, 150 is good, need to test higher vals
    t_crop = 0  # top, 100 is good. test higher vals
    return img_array[t_crop:665, h_crop:-h_crop]  # removes 150 pixels from each side, removes hood, and removes 100 pixels from top

def save_model(name):
    model.save('models/h5_models/{}.h5'.format(name))

os.chdir('S:/Git/traffic-lights/data/video')


x_train = []
for vid in os.listdir('./'):
    if '.png' in vid:
        x_train.append(crop_image(cv2.imread(vid)))

# x_train = np.array(x_train).flatten()
# with open('test', 'w') as f:
#     f.write('\n'.join([str(i) for i in x_train]))
# raise Exception
x_train = np.array([x_train], dtype=np.float32) / 255.
y_train = np.array([[0, 1, 0, 0]])

# W, H = 1164, 874
y_hood_crop = 665  # pixels from top where to crop image to get rid of hood.
video_shape = (5, 665, 814, 3)
labels = ['RED', 'GREEN', 'YELLOW', 'NONE']
proc_folder = 'data/.processed'

# reduction = 2

kernel_size = (3, 3)  # (3, 3)

model = Sequential()
model.add(TimeDistributed(Conv2D(8, kernel_size, activation='relu'), input_shape=video_shape))
model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3))))
# model.add(BatchNormalization())

model.add(TimeDistributed(Conv2D(12, kernel_size, activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3))))

model.add(TimeDistributed(Conv2D(16, kernel_size, activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3))))

model.add(TimeDistributed(Conv2D(24, kernel_size, activation='relu')))

model.add(TimeDistributed(Flatten()))
model.add(Flatten())

# model.add(CuDNNGRU(32, return_sequences=False))
# model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(len(labels), activation='softmax'))
print(model.summary())


# opt = keras.optimizers.RMSprop()
# opt = keras.optimizers.Adadelta()
# opt = keras.optimizers.Adagrad()
opt = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=1)
model.save('video_test.h5')


