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
from utils.data_generator import DataGenerator
from utils.eta_tool import ETATool
from utils.basedir import BASEDIR


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
# set_session(tf.Session(config=config))

os.chdir(BASEDIR)

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

    plt.imshow(traffic.BGR2RGB(img))
    plt.show()
    plt.title('Prediction: {}'.format(traffic.labels[pred]))
    plt.show()


def plot_features(layer_idx, img_class=None, img_idx=None, square=2, filter_idx=None):
    title_layer_idx = float(layer_idx)
    fig = plt.figure(0)
    fig.clf()
    layers = [layer for layer in traffic.model.layers if 'conv' in layer.name]
    layer_idx = [idx for idx, layer in enumerate(traffic.model.layers) if layer.name == layers[layer_idx].name][0]

    if img_idx is None:
        class_choice = random.choice(traffic.labels)
    else:
        class_choice = img_class
    x_test = get_img_paths('validation', class_choice)
    if img_idx is None:
        img_choice = random.choice(x_test)
    else:
        img_choice = x_test[img_idx]
    img = cv2.imread('{}/.validation/{}/{}'.format(traffic.proc_folder, class_choice, img_choice))
    img = (img / 255).astype(np.float32)
    print('Image index: {}'.format(x_test.index(img_choice)))

    plt.imshow(traffic.BGR2RGB(img))

    feature_model = Model(inputs=traffic.model.inputs, outputs=traffic.model.layers[layer_idx].output)
    feature_maps = feature_model.predict(np.array([img]))
    main_fig = plt.figure(1)
    main_fig.clf()

    if filter_idx is None:
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
        # ax.title.set_text('Conv2D Layer Index: {}'.format(title_layer_idx))
    else:
        plt.imshow(feature_maps[0, :, :, filter_idx], cmap='gray')
    plt.show()


def plot_features_old(layer_idx, square=2, img_class=None, img_idx=None):
    title_layer_idx = float(layer_idx)
    fig = plt.figure(0)
    fig.clf()
    layers = [layer for layer in traffic.model.layers if 'pool' in layer.name]
    layer_idx = [idx for idx, layer in enumerate(traffic.model.layers) if layer.name == layers[layer_idx].name][0]

    if img_idx is None:
        class_choice = random.choice(traffic.labels)
    else:
        class_choice = img_class
    x_test = get_img_paths('validation', class_choice)
    if img_idx is None:
        img_choice = random.choice(x_test)
    else:
        img_choice = x_test[img_idx]
    img = cv2.imread('{}/.validation/{}/{}'.format(traffic.proc_folder, class_choice, img_choice))
    img = (img / 255).astype(np.float32)
    print('Image index: {}'.format(x_test.index(img_choice)))

    plt.imshow(traffic.BGR2RGB(img))

    feature_model = Model(inputs=traffic.model.inputs, outputs=traffic.model.layers[layer_idx].output)
    feature_maps = feature_model.predict(np.array([img]))
    main_fig = plt.figure(1)
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
    plt.show()


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


class TrafficLightsModel:
    def __init__(self, force_reset=False):
        self.eta_tool = ETATool()
        # self.W, self.H = 1164, 874
        self.y_hood_crop = 665  # pixels from top where to crop image to get rid of hood.
        self.cropped_shape = (665, 814, 3)  # (515, 814, 3)
        self.labels = ['RED', 'GREEN', 'YELLOW', 'NONE']
        self.proc_folder = 'data/.processed'

        # self.reduction = 2
        self.batch_size = 32
        self.test_percentage = 0.2  # percentage of total data to be validated on
        self.num_flow_images = 3  # number of extra images to randomly generate per each input image
        self.dataloader_workers = 14  # used by keras to load input images, there is diminishing returns at high values (>~10)

        self.limit_samples = 3000

        self.model = None

        self.force_reset = force_reset
        self.num_train = 0
        self.num_valid = 0
        self.finished_file = 'data/.finished'
        self.class_weight = {}

        self.datagen_threads = 0
        self.datagen_max_threads = 30  # used to generate randomly transformed data (dependant on your CPU, set lower if it starts to freeze)

    def do_init(self):
        self.check_data()
        if self.needs_reset:
            self.reset_countdown()
            self.reset_data()
            self.process_images()
            self.create_val_images()  # create validation set for model

        # continue
        self.set_class_weight()
        train_gen, valid_gen = self.get_generators()
        return train_gen, valid_gen
        # self.train_batches(train_gen, valid_gen)


    def train_batches(self, train_generator, valid_generator, restart=False, epochs=50):
        if self.model is None or restart:
            self.model = self.get_model()

        # opt = keras.optimizers.RMSprop()
        # opt = keras.optimizers.Adadelta()
        # opt = keras.optimizers.Adagrad()
        opt = keras.optimizers.Adam(0.001*.4)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        self.model.fit_generator(train_generator,
                                 epochs=epochs,
                                 validation_data=valid_generator,
                                 workers=self.dataloader_workers,
                                 class_weight=self.class_weight)

    def get_model(self):
        # model = Sequential()
        # model.add(Dense(64, activation='relu', input_shape=(np.product(self.cropped_shape),)))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(4, activation='softmax'))
        # return model

        kernel_size = (3, 3)  # (3, 3)

        model = Sequential()
        model.add(Conv2D(12, kernel_size, activation='relu', input_shape=self.cropped_shape))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        # model.add(BatchNormalization())

        model.add(Conv2D(12, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Conv2D(16, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Conv2D(32, kernel_size, activation='relu'))


        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(len(self.labels), activation='softmax'))

        return model

    def BGR2RGB(self, arr):  # easier to inference on, EON uses BGR images
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def get_generators(self):
        train_dir = '{}/.train'.format(self.proc_folder)
        valid_dir = '{}/.validation'.format(self.proc_folder)
        train_generator = DataGenerator(train_dir, self.labels, self.batch_size)  # keeps data in BGR format and normalizes
        valid_generator = DataGenerator(valid_dir, self.labels, self.batch_size * 2)
        return train_generator, valid_generator

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

        open(self.finished_file, 'a').close()  # create finished file so we know in the future not to process data
        print('Finished, moving on to training!')

    def transform_and_crop_image(self, image_class, photo, datagen):
        self.datagen_threads += 1
        base_img = cv2.imread('data/{}/{}'.format(image_class, photo))  # loads uint8 BGR array
        imgs = np.array([base_img for _ in range(self.num_flow_images)])

        # randomly transform images
        try:
            batch = datagen.flow(imgs, batch_size=self.num_flow_images)[0]
        except:
            print(imgs)
            print(photo)
            raise Exception()
        flowed_imgs = [img.astype(np.uint8) for img in batch]  # convert from float32 0 to 255 to uint8 0 to 255

        flowed_imgs.append(base_img)  # append original non flowed image so we can crop and copy original cropped as well.
        cropped_imgs = [self.crop_image(img) for img in flowed_imgs]
        for k, img in enumerate(cropped_imgs):
            cv2.imwrite('{}/{}/{}.{}.png'.format(self.proc_folder, image_class, photo[:-4], k), img)
        self.datagen_threads -= 1

    def process_class(self, image_class, photos, datagen):  # manages processing threads
        t = time.time()
        self.eta_tool.init(t, len(photos))
        for idx, photo in enumerate(photos):
            self.eta_tool.log(idx, time.time())
            if time.time() - t > 15:
                # print('{}: Working on photo {} of {}.'.format(image_class, idx + 1, len(photos)))
                print('{}: Time to completion: {}'.format(image_class, self.eta_tool.get_eta))
                t = time.time()

            threading.Thread(target=self.transform_and_crop_image, args=(image_class, photo, datagen)).start()
            time.sleep(1 / 7.)  # spin up threads slightly slower
            while self.datagen_threads > self.datagen_max_threads:
                pass

        while self.datagen_threads != 0:  # wait for all threads to complete before continuing
            pass
        print('{}: Finished!'.format(image_class))

    def reset_countdown(self):
        if os.path.exists(self.proc_folder):  # don't show message if no data to delete
            print('WARNING: RESETTING PROCESSED DATA!', flush=True)
            print('This means all randomly transformed images will be erased and regenerated. Which may take some time depending on the amount of data you have.', flush=True)
            time.sleep(2)
            for i in range(8):
                sec = 8 - i
                multi = 's' if sec > 1 else ''  # gotta be grammatically correcet
                print('Resetting data in {} second{}!'.format(sec, multi))
                time.sleep(1.2)
            print('RESETTING DATA NOW', flush=True)

    def crop_image(self, img_array):
        h_crop = 175  # horizontal, 150 is good, need to test higher vals
        t_crop = 0  # top, 100 is good. test higher vals
        return img_array[t_crop:self.y_hood_crop, h_crop:-h_crop]  # removes 150 pixels from each side, removes hood, and removes 100 pixels from top

    def process_images(self):
        datagen = ImageDataGenerator(
                rotation_range=3.75,
                width_shift_range=0,
                height_shift_range=0,
                shear_range=0,
                zoom_range=0.15,
                horizontal_flip=False,  # todo: testing false
                fill_mode='nearest')

        print('Randomly transforming and cropping input images, please wait...'.format(len(self.labels)))
        print('Your system may slow until the process completes. Try reducing the max threads if it locks up.')
        for image_class in self.labels:
            photos = os.listdir('data/{}'.format(image_class))
            random.shuffle(photos)

            while len(photos) > self.limit_samples:
                del photos[random.randint(0, len(photos) - 1)]

            self.process_class(image_class, photos, datagen)
        print('All finished!')

    def reset_data(self):
        if os.path.exists(self.proc_folder):
            shutil.rmtree(self.proc_folder, ignore_errors=True)
        finished_file = self.finished_file
        if os.path.exists(finished_file):
            os.remove(finished_file)
        io_sleep()
        os.makedirs('{}/{}'.format(self.proc_folder, '.train'))
        os.mkdir('{}/{}'.format(self.proc_folder, '.validation'))
        for image_class in self.labels:
            os.mkdir('{}/{}'.format(self.proc_folder, image_class))
            os.mkdir('{}/.train/{}'.format(self.proc_folder, image_class))
            os.mkdir('{}/.validation/{}'.format(self.proc_folder, image_class))

    def set_class_weight(self):
        image_label_nums = {}
        for label in self.labels:
            image_label_nums[label] = len(os.listdir('{}/.train/{}'.format(self.proc_folder, label)))
            # self.num_train += len(os.listdir('{}/.train/{}'.format(self.proc_folder, cls)))
            # self.num_valid += len(os.listdir('{}/.validation/{}'.format(self.proc_folder, cls)))
        for label in self.labels:
            self.class_weight[self.labels.index(label)] = 1 / (image_label_nums[label] / max(image_label_nums.values()))  # get class weight. class with 50 samples and max 100 gets assigned 2.0
        tmp_prnt = {self.labels[cls]: self.class_weight[cls] for cls in self.class_weight}
        print('Class weights: {}'.format(tmp_prnt))

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
        return not os.path.exists(self.finished_file) or self.force_reset


traffic = TrafficLightsModel(force_reset=False)
train_gen, valid_gen = traffic.do_init()
if __name__ == '__main__':
    traffic.train_batches(train_gen, valid_gen)
