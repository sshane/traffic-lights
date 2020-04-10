import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
#
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2*1024)])
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNLSTM
from tensorflow.keras import Sequential

num_samples = 1000000
timesteps = 30
timestep_size = 4

x = np.random.rand(num_samples, timesteps, timestep_size)
y = np.random.rand(num_samples, 1)
print('x shape: {}'.format(x.shape))
print('y shape: {}'.format(y.shape))

# print('Press enter to start training...', flush=True)
# input()
model = Sequential()
model.add(CuDNNLSTM(128, return_sequences=True, input_shape=x.shape[1:]))
model.add(CuDNNLSTM(64, return_sequences=True))
model.add(CuDNNLSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, batch_size=256, epochs=500)
