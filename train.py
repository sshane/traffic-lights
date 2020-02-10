import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
from shutil import copyfile
import threading
import time
import shutil

os.chdir('C:/Git/traffic-lights')



def show_preds(choice=None, use_test=True):
    if choice is None:
        choice = random.randint(0, len(traffic.x_test) - 1)
    if use_test:
        img = traffic.x_test[choice]
    else:
        img = traffic.x_train[choice]

    pred = traffic.model.predict(np.array([img]))[0]
    pred = np.argmax(pred)

    plt.clf()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Prediction: {}'.format(traffic.classes[pred]))
    plt.show()

def plot_features(layer_idx, img_idx=0, square=2):
    title_layer_idx = float(layer_idx)
    fig = plt.figure(0)
    fig.clf()
    layers = [layer for layer in traffic.model.layers if 'pool' in layer.name]
    layer_idx = [idx for idx, layer in enumerate(traffic.model.layers) if layer.name == layers[layer_idx].name][0]

    img = traffic.x_test[img_idx]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    feature_model = Model(inputs=traffic.model.inputs, outputs=traffic.model.layers[layer_idx].output)
    feature_maps = feature_model.predict(np.array([img]))
    main_fig = plt.figure()
    main_fig.clf()

    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = main_fig.add_subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    ax.title.set_text('Conv2D Layer Index: {}'.format(title_layer_idx))
    # plt.show()

def plot_filters(layer_idx=0):
    layers = [layer for layer in traffic.model.layers if 'conv' in layer.name]
    layer_idx = [idx for idx, layer in enumerate(traffic.model.layers) if layer.name == layers[layer_idx].name][0]
    plt.clf()
    filters, biases = traffic.model.layers[layer_idx].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = traffic.model.layers[layer_idx].filters // 4, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()


class TrafficLightsModel:
    def __init__(self, force_reset=False):
        self.W, self.H = 1164, 874
        self.reduction = 2
        self.batch_size = 32
        self.classes = ['RED', 'GREEN', 'YELLOW', 'NONE']

        self.model = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.reset_flowed = force_reset
        self.img_gen_finished = 0.

    def do_init(self):
        if len(os.listdir('data/flowed')) < len(self.classes) or len(os.listdir('data/flowed/{}'.format(self.classes[0]))) == 0 or self.reset_flowed:
            self.reset_data()
            self.image_generator()
            while True:
                time.sleep(5)
                if self.img_gen_finished == len(self.classes):
                    self.load_imgs()
                    break
        else:
            self.load_imgs()

    def train(self):
        if self.model is None:
            self.model = self.get_model()
        opt = keras.optimizers.RMSprop()
        opt = keras.optimizers.Adadelta()
        opt = keras.optimizers.Adagrad()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                       epochs=100,
                       batch_size=self.batch_size,
                       validation_data=(self.x_test, self.y_test))

    def get_model(self):
        kernel_size = (2, 2)  # (3, 3)

        model = Sequential()
        model.add(Conv2D(16, kernel_size, activation='relu', input_shape=self.x_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(32, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(16, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(len(self.classes), activation='softmax'))

        return model

    def load_imgs(self):
        print('Loading and resizing data...', flush=True)
        for phot_class, phot_dir in enumerate(self.classes):
            for phot in os.listdir('data/flowed/{}'.format(phot_dir)):
                img = cv2.imread('data/flowed/{}/{}'.format(phot_dir, phot))
                img = cv2.resize(img, dsize=(self.W // self.reduction, self.H // self.reduction), interpolation=cv2.INTER_CUBIC)
                img = np.asarray(img) / 255
                self.x_train.append(img)
                self.y_train.append(self.one_hot(phot_class))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=0.2)
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)
        self.train()

    def reset_data(self):
        for photo_class in self.classes:
            if os.path.exists('data/flowed/{}'.format(photo_class)):
                shutil.rmtree('data/flowed/{}'.format(photo_class))
            time.sleep(0.1)
            os.makedirs('data/flowed/{}'.format(photo_class))

    def image_generator(self, num_images=10):
        datagen = ImageDataGenerator(
                rotation_range=2,
                width_shift_range=0,
                height_shift_range=0,
                shear_range=0,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

        print('Starting {} threads to randomly transform input images'.format(len(self.classes)))
        for image_class in self.classes:
            photos = os.listdir('data/{}'.format(image_class))
            threading.Thread(target=self.generate_images, args=(image_class, photos, datagen, num_images)).start()

    def generate_images(self, image_class, images, datagen, num_images):
        t = time.time()
        for idx, photo in enumerate(images):
            if time.time() - t > 10:
                print('{}: Working on photo {} of {}.'.format(image_class, idx + 1, len(images)))
                t = time.time()
            copyfile('data/{}/{}'.format(image_class, photo), 'data/flowed/{}/{}'.format(image_class, photo))  # copy original photo
            img = load_img('data/{}/{}'.format(image_class, photo))
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            for i, batch in enumerate(datagen.flow(x, batch_size=1, save_to_dir='data/flowed/{}'.format(image_class), save_prefix=image_class, save_format='png')):
                if i == num_images:
                    break
        self.img_gen_finished += 1
        print('{}: Finished!'.format(image_class))

    def one_hot(self, idx):
        one = [0] * len(self.classes)
        one[idx] = 1
        return one

    def save_model(self):
        self.model.save('models/h5_models/model.h5')


traffic = TrafficLightsModel(force_reset=False)
traffic.do_init()
