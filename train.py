import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
import threading
import time
import shutil
import pickle
from utils.custom_data_generator import CustomDataGenerator
from utils.eta_tool import ETATool
from utils.basedir import BASEDIR
from threading import Lock
import wandb
from wandb.keras import WandbCallback

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.chdir(BASEDIR)

def get_img_paths(typ, class_choice):
    return os.listdir('{}/.{}/{}'.format(traffic.proc_folder, typ, class_choice))


def show_preds():
    class_choice = random.choice(traffic.data_labels)
    x_test = get_img_paths('validation', class_choice)
    img_choice = random.choice(x_test)
    img = cv2.imread('{}/.validation/{}/{}'.format(traffic.proc_folder, class_choice, img_choice))
    img = (img / 255).astype(np.float32)

    prediction = traffic.model.predict(np.array([img]))[0]
    pred = np.argmax(prediction)

    plt.clf()

    plt.imshow(traffic.BGR2RGB(img))
    plt.show()
    if traffic.use_model_labels:
        labels = traffic.model_labels
    else:
        labels = traffic.data_labels
    plt.title('Prediction: {} ({}%)'.format(traffic.model_labels[pred], round(prediction[pred] * 100, 2)))
    plt.show()


def plot_features(layer_idx, img_class=None, img_idx=None, square=2, filter_idx=None):
    title_layer_idx = float(layer_idx)
    fig = plt.figure(0)
    fig.clf()
    layers = [layer for layer in traffic.model.layers if 'conv' in layer.name]
    layer_idx = [idx for idx, layer in enumerate(traffic.model.layers) if layer.name == layers[layer_idx].name][0]

    if img_idx is None:
        class_choice = random.choice(traffic.data_labels)
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
        class_choice = random.choice(traffic.data_labels)
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
    def __init__(self, cfg, force_reset=False):
        self.wandb_config = cfg
        self.eta_tool = ETATool()
        # self.W, self.H = 1164, 874
        self.y_hood_crop = 665  # pixels from top where to crop image to get rid of hood.
        self.cropped_shape = (665, 814, 3)  # (515, 814, 3)
        self.data_labels = ['RED', 'GREEN', 'YELLOW', 'NONE']

        self.transform_old_labels = {'RED': 'SLOW', 'GREEN': 'GREEN', 'YELLOW': 'SLOW', 'NONE': 'NONE'}
        self.model_labels = ['SLOW', 'GREEN', 'NONE']
        self.use_model_labels = True

        self.proc_folder = 'data/.processed'

        # self.batch_size = 36
        self.batch_size = self.wandb_config.batch_size
        self.test_percentage = 0.15  # percentage of total data to be validated on
        self.dataloader_workers = 256  # used by keras to load input images, there is diminishing returns at high values (>~10)

        self.max_samples_per_class = 14500  # unused after transformed data is created

        self.model = None

        self.force_reset = force_reset
        self.finished_file = 'data/.finished'
        self.class_weight = {}

        self.datagen_threads = 0
        self.datagen_max_threads = 128  # used to generate randomly transformed data (dependant on your CPU, set lower if it starts to freeze)
        self.num_flow_images = 5  # number of extra images to randomly generate per each input image
        self.lock = Lock()

    def do_init(self):
        self.check_data()
        if self.needs_reset:
            self.reset_countdown()
            self.reset_data()
            self.create_validation_set()  # create validation set for model
            self.transform_images()

        self.set_class_weight()
        train_gen, valid_gen = self.get_generators()
        return train_gen, valid_gen

    def train_batches(self, train_generator, valid_generator, restart=False, epochs=50):
        if not restart:
            print('Want to load a previous model to continue training?')
            load_prev = input('[Y/n]: ').lower().strip()
            if load_prev in ['y', 'yes']:
                model_name = input('Enter the h5 model name without the .h5 suffix: ')
                self.model = keras.models.load_model('models/h5_models/{}.h5'.format(model_name))
                print('SUCCESSFULLY LOADED {}.h5'.format(model_name))
            elif self.model is None or restart:
                self.model = self.get_model_2()
        else:
            self.model = self.get_model_wandb()

        # opt = keras.optimizers.RMSprop()
        # opt = keras.optimizers.Adadelta()
        # opt = keras.optimizers.Adagrad()
        # opt = keras.optimizers.Adam(0.001*.4)
        opt = keras.optimizers.Adam(lr=self.wandb_config.learning_rate, amsgrad=True)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['acc'])


        class_weight = self.class_weight if self.wandb_config.use_class_weight else None
        filepath = "models/h5_models/optimized_model/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
        self.model.fit(train_generator,
                       epochs=epochs,
                       validation_data=valid_generator,
                       workers=self.dataloader_workers,
                       callbacks=[WandbCallback(), ModelCheckpoint(filepath, save_best_only=False, save_weights_only=False)],
                       class_weight=class_weight)

    def get_model_1(self):
        # model = Sequential()
        # model.add(Dense(64, activation='relu', input_shape=(np.product(self.cropped_shape),)))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(4, activation='softmax'))
        # return model

        kernel_size = (3, 3)  # almost no effect on model size

        print('USING NEW MODEL')
        model = Sequential()
        model.add(Conv2D(12, kernel_size, strides=1, activation='relu', input_shape=self.cropped_shape))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        # model.add(BatchNormalization())

        model.add(Conv2D(24, kernel_size, strides=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size, strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        if not self.use_model_labels:
            model.add(Dense(len(self.data_labels), activation='softmax'))
        else:
            model.add(Dense(len(self.model_labels), activation='softmax'))
        return model

    def get_model_2(self):
        # model = Sequential()
        # model.add(Dense(64, activation='relu', input_shape=(np.product(self.cropped_shape),)))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(4, activation='softmax'))
        # return model
        # model.add(Dropout(0.3))

        kernel_size = (3, 3)  # (3, 3)

        model = Sequential()
        model.add(Conv2D(6, kernel_size, activation='relu', input_shape=self.cropped_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(BatchNormalization())

        model.add(Conv2D(12, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(36, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        print('USING OLD MODEL')

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.3))
        if not self.use_model_labels:
            model.add(Dense(len(self.data_labels), activation='softmax'))
        else:
            model.add(Dense(len(self.model_labels), activation='softmax'))
        return model

    def BGR2RGB(self, arr):  # easier to inference on, EON uses BGR images
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def get_generators(self):
        train_dir = '{}/.train'.format(self.proc_folder)
        valid_dir = '{}/.validation'.format(self.proc_folder)
        train_generator = CustomDataGenerator(train_dir, self.data_labels, self.model_labels, self.transform_old_labels, self.use_model_labels, self.batch_size)  # keeps data in BGR format and normalizes
        valid_generator = CustomDataGenerator(valid_dir, self.data_labels, self.model_labels, self.transform_old_labels, self.use_model_labels, self.batch_size)
        return train_generator, valid_generator

    def create_validation_set(self):
        print('Your system may slow until the process completes. Try reducing `self.datagen_max_threads` if it locks up.')
        print('Do NOT delete anything in the `.processed` folder while it\'s working.\n')
        print('Creating train and validation sets!', flush=True)
        for idx, image_class in enumerate(self.data_labels):  # load all image names and class
            print('Working on class: {}'.format(image_class))
            class_dir = 'data/{}'.format(image_class)
            images = [{'img_path': '{}/{}'.format(class_dir, img),
                       'img_name': img} for img in os.listdir(class_dir)]

            random.shuffle(images)
            while len(images) > self.max_samples_per_class:  # only keep up to max samples
                del images[random.randint(0, len(images) - 1)]

            train, valid = train_test_split(images, test_size=self.test_percentage)  # split by class

            for img in train:
                shutil.copyfile(img['img_path'], '{}/.train_temp/{}/{}'.format(self.proc_folder, image_class, img['img_name']))
            for img in valid:
                shutil.copyfile(img['img_path'], '{}/.validation_temp/{}/{}'.format(self.proc_folder, image_class, img['img_name']))
        print()

    def transform_and_crop_image(self, image_class, photo_path, datagen, is_train):
        with self.lock:
            self.datagen_threads += 1
        original_img = cv2.imread(photo_path)  # loads uint8 BGR array
        flowed_imgs = []
        if is_train:  # don't transform validation images
            imgs = np.array([original_img for _ in range(self.num_flow_images)])
            # randomly transform images
            try:
                batch = datagen.flow(imgs, batch_size=self.num_flow_images)[0]
            except Exception as e:
                print(imgs)
                print(photo_path)
                raise Exception('Error in transform_and_crop_image: {}'.format(e))
            flowed_imgs = [img.astype(np.uint8) for img in batch]  # convert from float32 0 to 255 to uint8 0 to 255

        flowed_imgs.append(original_img)  # append original non flowed image so we can crop and copy original as well
        cropped_imgs = [self.crop_image(img) for img in flowed_imgs]
        for idx, img in enumerate(cropped_imgs):
            photo_name = photo_path.split('/')[-1][:-4]  # get name from path excluding extension
            # print('{}/.train/{}/{}'.format(self.proc_folder, image_class, photo_name))
            if is_train:
                cv2.imwrite('{}/.train/{}/{}.{}.png'.format(self.proc_folder, image_class, photo_name, idx), img)
            else:
                cv2.imwrite('{}/.validation/{}/{}.{}.png'.format(self.proc_folder, image_class, photo_name, idx), img)
        with self.lock:
            self.datagen_threads -= 1

    def process_class(self, image_class, photo_paths, datagen, is_train):  # manages processing threads
        t = time.time()
        self.eta_tool.init(t, len(photo_paths))
        train_msg = 'train' if is_train else 'valid'
        for idx, photo_path in enumerate(photo_paths):
            self.eta_tool.log(idx, time.time())
            if time.time() - t > 15:
                # print('{}: Working on photo {} of {}.'.format(image_class, idx + 1, len(photos)))
                print('{} ({}): Time to completion: {}'.format(image_class, train_msg, self.eta_tool.get_eta))
                t = time.time()

            threading.Thread(target=self.transform_and_crop_image, args=(image_class, photo_path, datagen, is_train)).start()
            while self.datagen_threads > self.datagen_max_threads:
                time.sleep(1 / 5.)

        while self.datagen_threads != 0:  # wait for all threads to complete before continuing
            time.sleep(1)

        print('{} ({}): Finished!'.format(image_class, train_msg))

    def reset_countdown(self):
        if os.path.exists(self.proc_folder):  # don't show message if no data to delete
            print('WARNING: RESETTING PROCESSED DATA!', flush=True)
            print('This means all randomly transformed images will be erased and regenerated. '
                  'Which may take some time depending on the amount of data you have.', flush=True)
            time.sleep(2)
            for i in range(10):
                sec = 10 - i
                multi = 's' if sec > 1 else ''  # gotta be grammatically correcet
                print('Resetting data in {} second{}!'.format(sec, multi))
                time.sleep(1)
            print('RESETTING DATA NOW', flush=True)

    def crop_image(self, img_array):
        h_crop = 175  # horizontal, 150 is good, need to test higher vals
        t_crop = 0  # top, 100 is good. test higher vals
        return img_array[t_crop:self.y_hood_crop, h_crop:-h_crop]  # removes 150 pixels from each side, removes hood, and removes 100 pixels from top

    def transform_images(self):
        datagen = ImageDataGenerator(
            rotation_range=2.5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0,
            zoom_range=0.1,
            horizontal_flip=True,  # todo: testing false
            fill_mode='nearest')

        print('Randomly transforming and cropping input images, please wait...')
        for image_class in self.data_labels:
            photos_train = os.listdir('{}/.train_temp/{}'.format(self.proc_folder, image_class))
            photos_valid = os.listdir('{}/.validation_temp/{}'.format(self.proc_folder, image_class))

            photos_train = ['{}/.train_temp/{}/{}'.format(self.proc_folder, image_class, img) for img in photos_train]  # adds path
            photos_valid = ['{}/.validation_temp/{}/{}'.format(self.proc_folder, image_class, img) for img in photos_valid]

            self.process_class(image_class, photos_train, datagen, True)
            # print('starting cropping of validation data')
            self.process_class(image_class, photos_valid, datagen, False)  # no transformations, only crop for valid
        shutil.rmtree('{}/.train_temp'.format(self.proc_folder))
        shutil.rmtree('{}/.validation_temp'.format(self.proc_folder))

        open(self.finished_file, 'a').close()  # create finished file so we know in the future not to process data
        print('All finished, moving on to training!')

    def reset_data(self):
        if os.path.exists(self.proc_folder):
            shutil.rmtree(self.proc_folder, ignore_errors=True)

        if os.path.exists(self.finished_file):
            os.remove(self.finished_file)
        io_sleep()

        os.mkdir(self.proc_folder)
        for image_class in self.data_labels:
            # os.mkdir('{}/{}'.format(self.proc_folder, image_class))
            os.makedirs('{}/.train/{}'.format(self.proc_folder, image_class))
            os.makedirs('{}/.train_temp/{}'.format(self.proc_folder, image_class))
            os.makedirs('{}/.validation/{}'.format(self.proc_folder, image_class))
            os.makedirs('{}/.validation_temp/{}'.format(self.proc_folder, image_class))

    def set_class_weight(self):
        if not self.use_model_labels:
            labels = self.data_labels
            label_img_count = {}
            for label in self.data_labels:
                label_img_count[label] = len(os.listdir('{}/.train/{}'.format(self.proc_folder, label)))
        else:
            labels = self.model_labels
            label_img_count = {lbl: 0 for lbl in self.model_labels}
            for label in self.data_labels:
                model_label = self.transform_old_labels[label]
                label_img_count[model_label] += len(os.listdir('{}/.train/{}'.format(self.proc_folder, label)))

        for label in label_img_count:
            self.class_weight[labels.index(label)] = 1 / (label_img_count[label] / max(label_img_count.values()))  # get class weight. class with 50 samples and max 100 gets assigned 2.0

        tmp_prnt = {labels[cls]: self.class_weight[cls] for cls in self.class_weight}
        print('Class weights: {}'.format(tmp_prnt))

    def one_hot(self, idx):
        if not self.use_model_labels:
            one = [0] * len(self.data_labels)
        else:
            one = [0] * len(self.model_labels)
        one[idx] = 1
        return one

    def check_data(self):
        if not os.path.exists('data'):
            print('DATA DIRECTORY DOESN\'T EXIST!')
            os.mkdir('data')
            raise Exception('Please unzip the data.zip archive into data directory')
        data_files = os.listdir('data')
        if not all([i in data_files for i in self.data_labels]):
            raise Exception('Please unzip the data.zip archive into data directory')

    @property
    def needs_reset(self):
        return not os.path.exists(self.finished_file) or not os.path.exists(self.proc_folder) or self.force_reset

    def get_model_wandb(self):
        kernel_size = (self.wandb_config.kernel_size, self.wandb_config.kernel_size)

        model = Sequential()
        pool_1 = self.wandb_config.pool_1
        model.add(Conv2D(self.wandb_config.conv_1_nodes, kernel_size, activation='relu', input_shape=self.cropped_shape))
        model.add(MaxPooling2D(pool_size=(pool_1, pool_1)))

        model.add(Conv2D(self.wandb_config.conv_2_nodes, kernel_size, activation='relu'))
        pool_2 = self.wandb_config.pool_2
        model.add(MaxPooling2D(pool_size=(pool_2, pool_2)))

        model.add(Conv2D(self.wandb_config.conv_3_nodes, kernel_size, activation='relu'))
        pool_3 = self.wandb_config.pool_3
        model.add(MaxPooling2D(pool_size=(pool_3, pool_3)))

        # model.add(Conv2D(self.wandb_config.conv_4_nodes, kernel_size, activation='relu'))
        # model.add(MaxPooling2D(pool_size=(3, 3)))
        print('USING WANDB MODEL')

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(self.wandb_config.dense_1_nodes, activation='relu'))
        model.add(Dense(self.wandb_config.dense_2_nodes, activation='relu'))
        if not self.use_model_labels:
            model.add(Dense(len(self.data_labels), activation='softmax'))
        else:
            model.add(Dense(len(self.model_labels), activation='softmax'))
        return model


hyperparameter_defaults = dict(
    kernel_size=6,
    learning_rate=0.0005,
    batch_size=32,
    use_class_weight=True,

    conv_1_nodes=10,
    pool_1=4,
    conv_2_nodes=29,
    pool_2=3,
    conv_3_nodes=30,
    pool_3=3,

    dense_1_nodes=39,
    dense_2_nodes=182,
)


wandb.init(project="traffic-lights", config=hyperparameter_defaults)
wandb_config = wandb.config


traffic = TrafficLightsModel(wandb_config, force_reset=False)
train_gen, valid_gen = traffic.do_init()
if __name__ == '__main__':
    traffic.train_batches(train_gen, valid_gen, epochs=100, restart=True)
