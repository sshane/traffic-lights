import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential

num_samples = 250
h = 665
w = 814
c = 3

x = np.random.rand(num_samples, h, w, c)
y = [[0] * 4 for _ in range(num_samples)]
for i in range(len(y)):
    y[i][random.randint(0, 3)] = 1
y = np.array(y)

kernel_size = (3, 3)
model = Sequential()
model.add(Conv2D(12, kernel_size, strides=1, activation='relu', input_shape=x.shape[1:]))

model.add(Conv2D(24, kernel_size, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(48, kernel_size, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, kernel_size, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(12, kernel_size, strides=1, activation='relu'))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, batch_size=128, epochs=500)
