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
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
import threading
import time
import shutil
import pickle
from data_generator import DataGenerator


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
# set_session(tf.Session(config=config))

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)  # todo: ensure this leads to traffic-lights home directory


def get_img_paths(typ, class_choice):
    return os.listdir('{}/.{}/{}'.format(traffic.proc_folder, typ, class_choice))


def show_preds():
    class_choice = random.choice(traffic.labels)
    x_test = get_img_paths('validation', class_choice)
    img_choice = random.choice(x_test)
    img = cv2.imread('{}/.validation/{}/{}'.format(traffic.proc_folder, class_choice, img_choice))
    img = (img / 255).astype(np.float32)

    pred = traffic.model.predict(np.array([img]))[0]
    pred = np.argmax(pred)

    plt.clf()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.title('Prediction: {}'.format(traffic.labels[pred]))
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


def io_sleep():
    time.sleep(0.1)


def save_model(name):
    traffic.model.save('models/h5_models/{}.h5'.format(name))



class TrafficLightsModel:  # TODO: USE KERAS IMAGE LOADER
    def __init__(self, force_reset=False):
        self.W, self.H = 1164, 874
        self.y_hood_crop = 665  # pixels from top where to crop image to get rid of hood. this is perfect for smallish vehicles
        self.labels = ['RED', 'GREEN', 'YELLOW', 'NONE']
        self.proc_folder = 'data/.processed'

        self.reduction = 2
        self.batch_size = 32
        self.test_percentage = 0.2  # percentage of total data to be validated on
        self.num_flow_images = 5

        self.limit_samples = 500

        self.model = None
        # self.x_train = []
        # self.y_train = []
        # self.x_test = []
        # self.y_test = []

        self.reset_proc = force_reset
        self.classes_processed = 0
        self.num_train = 0
        self.num_valid = 0

    def do_init(self):
        self.check_data()
        if self.needs_reset:
            self.reset_data()
            self.process_images()
            while True:
                time.sleep(5)  # wait for background threads to finish processing images
                if self.classes_processed == len(self.labels):
                    self.create_val_images()  # create validation set for model
                    # self.store_arrays()
                    break
        # continue
        self.set_num_images()
        train_gen, valid_gen = self.get_generators()
        return train_gen, valid_gen
        # self.train_batches(train_gen, valid_gen)
        # self.train()

    def store_arrays(self):
        for label in self.labels:
            print('Converting {} images to arrays for faster loading...'.format(label))
            label_path = '{}/{{}}/{}'.format(self.proc_folder, label)
            for image in os.listdir(label_path.format('.train')):
                img_path = '{}/{}'.format(label_path.format('.train'), image)
                img_array = cv2.imread(img_path).astype(np.float32) / 255.
                new_img_path = label_path.format('.train_pickle')
                new_img_name = '.'.join(image.split('/')[-1].split('.')[:-1])
                new_img_path = '{}/{}.p'.format(new_img_path, new_img_name)
                h5f = h5py.File(new_img_path, 'w')
                h5f.create_dataset('dataset1', data=img_array)
                # np.savez_compressed(new_img_path, *[img_array for _ in range(128)])
                print('done')
                # with open(new_img_path, 'wb') as f:
                #     pickle.dump(img_array, f)
            for image in os.listdir(label_path.format('.validation')):
                img_path = '{}/{}'.format(label_path.format('.validation'), image)
                img_array = cv2.imread(img_path).astype(np.float32) / 255.
                new_img_path = label_path.format('.validation_pickle')
                new_img_name = '.'.join(image.split('/')[-1].split('.')[:-1])
                new_img_path = '{}/{}.p'.format(new_img_path, new_img_name)
                with open(new_img_path, 'wb') as f:
                    pickle.dump(img_array, f)


    def train_batches(self, train_generator, valid_generator, restart=False):
        if self.model is None or restart:
            self.model = self.get_model()

        # opt = keras.optimizers.RMSprop()
        # opt = keras.optimizers.Adadelta()
        # opt = keras.optimizers.Adagrad()
        opt = keras.optimizers.Adam()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        # train_steps = int(np.ceil(self.num_train / self.batch_size))  # calculated by keras with datagenerator now
        # valid_steps = int(np.ceil(self.num_valid / self.batch_size))

        self.model.fit_generator(train_generator,
                                 epochs=100,
                                 validation_data=valid_generator,
                                 workers=8)


    # def train(self):
    #     if self.model is None:
    #         self.model = self.get_model()
    #     # opt = keras.optimizers.RMSprop()
    #     # opt = keras.optimizers.Adadelta()
    #     # opt = keras.optimizers.Adagrad()
    #     opt = keras.optimizers.Adam()
    #
    #     self.model.compile(loss='categorical_crossentropy',
    #                        optimizer=opt,
    #                        metrics=['accuracy'])
    #
    #     self.model.fit(self.x_train, self.y_train,
    #                    epochs=100,
    #                    batch_size=self.batch_size,
    #                    validation_data=(self.x_test, self.y_test))


    def get_model(self):
        kernel_size = (2, 2)  # (3, 3)

        model = Sequential()
        model.add(Conv2D(16, kernel_size, activation='relu', input_shape=(self.y_hood_crop, self.W, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(16, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(len(self.labels), activation='softmax'))

        return model

    def RGB2BGR(self, arr):  # easier to inference on, EON uses BGR images
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return K.cast_to_floatx(rgb)

    def get_generators(self):
        train_dir = '{}/.train'.format(self.proc_folder)
        valid_dir = '{}/.validation'.format(self.proc_folder)
        train_generator = DataGenerator(train_dir, self.labels, self.batch_size)  # keeps data in BGR format and normalizes
        valid_generator = DataGenerator(valid_dir, self.labels, self.batch_size)
        return train_generator, valid_generator

    # def get_generators(self):
    #     train_dir = '{}/.train'.format(self.proc_folder)
    #     valid_dir = '{}/.validation'.format(self.proc_folder)
    #     train_datagen = ImageDataGenerator(rescale=1./255)  # modified line 1248 to convert RGB to BGR
    #     valid_datagen = ImageDataGenerator(rescale=1./255)
    #
    #     train_generator = train_datagen.flow_from_directory(
    #         train_dir,
    #         target_size=(self.y_hood_crop, self.W),  # height, width
    #         batch_size=self.batch_size,
    #         class_mode='categorical')
    #
    #     valid_generator = valid_datagen.flow_from_directory(
    #         valid_dir,
    #         target_size=(self.y_hood_crop, self.W),
    #         batch_size=self.batch_size,
    #         class_mode='categorical')
    #     return train_generator, valid_generator

    # def load_imgs(self):
    #     print('Loading data...', flush=True)
    #     for image_class, phot_dir in enumerate(self.classes):
    #         t = 0
    #         images = os.listdir('{}/{}'.format(self.proc_folder, phot_dir))
    #         for idx, phot in enumerate(images):
    #             if time.time() - t >= 5:
    #                 print('{}: Loading photo {} of {}.'.format(self.classes[image_class], idx + 1, len(images)))
    #                 t = time.time()
    #             img = cv2.imread('{}/{}/{}'.format(self.proc_folder, phot_dir, phot))
    #             # img = cv2.resize(img, dsize=(self.W // self.reduction, self.H // self.reduction), interpolation=cv2.INTER_CUBIC)  # don't resize
    #             try:
    #                 img = img.astype(np.float16) / 255  # normalize
    #                 self.x_train.append(img)
    #                 self.y_train.append(self.one_hot(image_class))
    #             except Exception as e:
    #                 print(e)
    #
    #     print('\nGetting train and test set, please wait...', flush=True)
    #     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=self.test_percentage)
    #     self.x_train = np.array(self.x_train)
    #     self.y_train = np.array(self.y_train)
    #     self.x_test = np.array(self.x_test)
    #     self.y_test = np.array(self.y_test)

    def create_val_images(self):
        print('Separating validation images!', flush=True)
        images = []
        for idx, image_class in enumerate(self.labels):  # load all image names and class
            class_dir = '{}/{}'.format(self.proc_folder, image_class)
            for image in os.listdir(class_dir):
                images.append({'path': '{}/{}'.format(class_dir, image), 'class': image_class})

        train, valid = train_test_split(images, test_size=self.test_percentage)
        for sample in train:
            img_name = sample['path'].split('/')[-1]
            shutil.move(sample['path'], '{}/.train/{}/{}'.format(self.proc_folder, sample['class'], img_name))
        for sample in valid:
            img_name = sample['path'].split('/')[-1]
            shutil.move(sample['path'], '{}/.validation/{}/{}'.format(self.proc_folder, sample['class'], img_name))

        for image_class in self.labels:
            io_sleep()
            os.rmdir('{}/{}'.format(self.proc_folder, image_class))  # should be empty, throw error if not

        open('data/.finished', 'a').close()  # create finished file so we know in the future not to process data
        print('Finished, moving on to training!')

    def flow_and_crop(self, image_class, photos, datagen):
        t = time.time()
        for idx, photo in enumerate(photos):
            if time.time() - t > 10:
                print('{}: Working on photo {} of {}.'.format(image_class, idx + 1, len(photos)))
                t = time.time()

            base_img = cv2.imread('data/{}/{}'.format(image_class, photo))  # loads uint8 BGR array
            imgs = np.array([base_img for _ in range(self.num_flow_images)])

            # randomly transform images
            batch = datagen.flow(imgs, batch_size=self.num_flow_images)[0]
            flowed_imgs = [img.astype(np.uint8) for img in batch]  # convert from float32 0 to 255 to uint8 0 to 255

            flowed_imgs.append(base_img)  # append original non flowed image so we can crop and copy original cropped as well.
            cropped_imgs = [img[0:self.y_hood_crop, 0:self.W] for img in flowed_imgs]  # crop out hood
            for k, img in enumerate(cropped_imgs):
                cv2.imwrite('{}/{}/{}.{}.png'.format(self.proc_folder, image_class, photo[:-4], k), img)

        self.classes_processed += 1
        if self.classes_processed != 4:
            print('{}: Finished!'.format(image_class))
        else:
            print('All finished!')


    def process_images(self):
        datagen = ImageDataGenerator(
                rotation_range=3.5,
                width_shift_range=0,
                height_shift_range=0,
                shear_range=0,
                zoom_range=0.18,
                horizontal_flip=True,
                fill_mode='nearest')

        print('Starting {} threads to randomly transform and crop input images, please wait...'.format(len(self.labels)))
        for image_class in self.labels:
            photos = os.listdir('data/{}'.format(image_class))
            while len(photos) > self.limit_samples:
                del photos[random.randint(0, len(photos) - 1)]
            threading.Thread(target=self.flow_and_crop, args=(image_class, photos, datagen)).start()

    def reset_data(self):
        if os.path.exists(self.proc_folder):
            shutil.rmtree(self.proc_folder, ignore_errors=True)

        io_sleep()
        os.makedirs('{}/{}'.format(self.proc_folder, '.train'))
        os.mkdir('{}/{}'.format(self.proc_folder, '.validation'))
        for image_class in self.labels:
            os.mkdir('{}/{}'.format(self.proc_folder, image_class))
            os.mkdir('{}/.train/{}'.format(self.proc_folder, image_class))
            os.mkdir('{}/.validation/{}'.format(self.proc_folder, image_class))

    def set_num_images(self):
        for cls in self.labels:
            self.num_train += len(os.listdir('{}/.train/{}'.format(self.proc_folder, cls)))
            self.num_valid += len(os.listdir('{}/.validation/{}'.format(self.proc_folder, cls)))

    def one_hot(self, idx):
        one = [0] * len(self.labels)
        one[idx] = 1
        return one

    def check_data(self):
        if not os.path.exists('data'):
            print('DATA DIRECTORY DOESN\'T EXIST!')
            os.mkdir('data')
            raise Exception('Please unzip the data.zip archive into data directory')
        data_files = os.listdir('data')
        if not all([i in data_files for i in self.labels]):
            raise Exception('Please unzip the data.zip archive into data directory')

    @property
    def needs_reset(self):
        return not os.path.exists('data/.finished')


traffic = TrafficLightsModel(force_reset=False)  # todo: set force_reset to reset cropped data
train_gen, valid_gen = traffic.do_init()
if __name__ == '__main__':
    traffic.train_batches(train_gen, valid_gen)