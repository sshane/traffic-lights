import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
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
import pickle

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


os.chdir(os.path.dirname(os.path.realpath(__file__)))  # todo: ensure this leads to traffic-lights home directory


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
    plt.show()
    plt.title('Prediction: {}'.format(traffic.classes[pred]))
    plt.show()


def plot_features(layer_idx, img_idx=0, square=2):
    title_layer_idx = float(layer_idx)
    fig = plt.figure(0)
    fig.clf()
    layers = [layer for layer in traffic.model.layers if 'pool' in layer.name]
    layer_idx = [idx for idx, layer in enumerate(traffic.model.layers) if layer.name == layers[layer_idx].name][0]

    img = traffic.x_test[img_idx]

    plt.imshow(cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB))

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
        self.y_hood_crop = 665  # pixels from top where to crop image to get rid of hood. this is perfect for smallish vehicles
        self.reduction = 2
        self.batch_size = 4
        self.classes = ['RED', 'GREEN', 'YELLOW', 'NONE']
        self.flow_images = 3  # 10 new images from 1 original image
        self.test_percentage = 0.2  # percentage of total data to be validated on

        self.model = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.reset_flowed = force_reset
        self.img_gen_finished = 0.

    def do_init(self):
        self.check_data()

        if len(os.listdir('data/flowed')) < len(self.classes) or len(os.listdir('data/flowed/{}'.format(self.classes[0]))) == 0 or self.reset_flowed:
            self.reset_data()
            # self.image_generator()
            self.crop_images()
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
        # opt = keras.optimizers.RMSprop()
        # opt = keras.optimizers.Adadelta()
        # opt = keras.optimizers.Adagrad()
        opt = keras.optimizers.Adam()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                       epochs=100,
                       batch_size=self.batch_size,
                       validation_data=(self.x_test, self.y_test))

    def get_model(self):
        kernel_size = (3, 3)  # (3, 3)

        model = Sequential()
        model.add(Conv2D(16, kernel_size, activation='relu', input_shape=self.x_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(32, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(32, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(len(self.classes), activation='softmax'))

        return model

    def load_imgs(self):
        print('Loading data...', flush=True)
        for phot_class, phot_dir in enumerate(self.classes):
            t = 0
            images = os.listdir('data/flowed/{}'.format(phot_dir))
            for idx, phot in enumerate(images):
                if time.time() - t >= 5:
                    print('{}: Loading photo {} of {}.'.format(self.classes[phot_class], idx + 1, len(images)))
                    t = time.time()
                img = cv2.imread('data/flowed/{}/{}'.format(phot_dir, phot))
                # img = cv2.resize(img, dsize=(self.W // self.reduction, self.H // self.reduction), interpolation=cv2.INTER_CUBIC)  # don't resize

                img = np.asarray(img, dtype=np.float32) / 255  # normalize
                if img.dtype == np.float32:
                    self.x_train.append(img)
                    self.y_train.append(self.one_hot(phot_class))

        print('\nGetting train and test set, please wait...', flush=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=self.test_percentage)
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)
        self.train()

    def load_imgs_old(self):
        print('Loading and resizing data...', flush=True)
        for phot_class, phot_dir in enumerate(self.classes):
            for phot in os.listdir('data/flowed/{}'.format(phot_dir)):
                img = cv2.imread('data/flowed/{}/{}'.format(phot_dir, phot))
                # img = cv2.resize(img, dsize=(self.W // self.reduction, self.H // self.reduction), interpolation=cv2.INTER_CUBIC)  # don't resize
                img = img[0:self.y_hood_crop, 0:self.W]  # full resolution but cropping out hood

                img = np.asarray(img) / 255  # normalize
                self.x_train.append(img)
                self.y_train.append(self.one_hot(phot_class))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=self.test_percentage)
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
            time.sleep(0.1)

    def crop_images(self):
        def crop_imgs(image_class, images):
            t = time.time()
            for idx, photo in enumerate(images):
                if time.time() - t > 10:
                    print('{}: Working on photo {} of {}.'.format(image_class, idx + 1, len(images)))
                    t = time.time()
                # copyfile('data/{}/{}'.format(image_class, photo), 'data/flowed/{}/{}'.format(image_class, photo))  # copy original photo
                # img = load_img('data/{}/{}'.format(image_class, photo))
                img = cv2.imread('data/{}/{}'.format(image_class, photo))
                img = np.asarray(img, dtype=np.float32)

                img = img[0:self.y_hood_crop, 0:self.W]

                cv2.imwrite('data/flowed/{}/{}.png'.format(image_class, photo[:-4]), img)
                # with open('data/flowed/{}/{}.pickle'.format(image_class, photo[:-4]), 'wb') as f:
                #     f.write(img)

            self.img_gen_finished += 1
            if self.img_gen_finished != 4:
                print('{}: Finished!'.format(image_class))
            else:
                print('All finished!')

        print('Starting {} threads to randomly transform input images, please wait...'.format(len(self.classes)))
        for image_class in self.classes:
            photos = os.listdir('data/{}'.format(image_class))
            threading.Thread(target=crop_imgs, args=(image_class, photos)).start()

    def image_generator(self):
        datagen = ImageDataGenerator(
                rotation_range=2,
                width_shift_range=0,
                height_shift_range=0,
                shear_range=0,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')

        print('Starting {} threads to randomly transform input images, please wait...'.format(len(self.classes)))
        for image_class in self.classes:
            photos = os.listdir('data/{}'.format(image_class))
            threading.Thread(target=self.generate_images, args=(image_class, photos, datagen)).start()

    def generate_images(self, image_class, images, datagen):
        t = time.time()
        for idx, photo in enumerate(images):
            if time.time() - t > 10:
                print('{}: Working on photo {} of {}.'.format(image_class, idx + 1, len(images)))
                t = time.time()
            # copyfile('data/{}/{}'.format(image_class, photo), 'data/flowed/{}/{}'.format(image_class, photo))  # copy original photo
            # img = load_img('data/{}/{}'.format(image_class, photo))
            img = cv2.imread('data/{}/{}'.format(image_class, photo))
            img = np.asarray(img, dtype=np.float32)
            img = img[0:self.y_hood_crop, 0:self.W]

            cv2.imwrite('data/flowed/{}'.format(image_class), img)
            # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            # for i, batch in enumerate(datagen.flow(x, batch_size=4, save_to_dir='data/flowed/{}'.format(image_class), save_prefix=image_class, save_format='png')):
            #     if i == self.flow_images:
            #         break
        self.img_gen_finished += 1
        if self.img_gen_finished != 4:
            print('{}: Finished!'.format(image_class))
        else:
            print('All finished!')

    def one_hot(self, idx):
        one = [0] * len(self.classes)
        one[idx] = 1
        return one

    def save_model(self, name):
        self.model.save('models/h5_models/{}.h5'.format(name))

    def check_data(self):
        if not os.path.exists('data'):
            print('DATA DIRECTORY DOESN\'T EXIST!')
            os.mkdir('data')
            raise Exception('Please unzip the data.zip archive into data directory')
        if not os.path.exists('data/flowed'):
            print('CREATING FLOWED DIRECTORY')
            os.mkdir('data/flowed')
        data_files = os.listdir('data')
        if not all([i in data_files for i in self.classes]):
            raise Exception('Please unzip the data.zip archive into data directory')

traffic = TrafficLightsModel(force_reset=False)  # todo: set force_reset to reset flowed data
traffic.do_init()
