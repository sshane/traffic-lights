import keras
import os
import random
import numpy as np
import cv2


class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, directory, labels, new_labels, transform_old_labels, use_new_labels, batch_size, shuffle=True):
        self.directory = directory
        self.labels = labels
        self.new_labels = new_labels
        self.transform_old_labels = transform_old_labels
        self.use_new_labels = use_new_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_paths, self.image_labels = self.get_files()

        self.shuffle_images()
        self.n = 0
        self.max = self.__len__()

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.image_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.array([self.load_image(file_name) for file_name in batch_x])
        y = np.array(batch_y)
        return x, y

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.shuffle_images()

    def get_files(self):
        image_paths = []
        image_labels = []
        for label in self.labels:
            label_path = '{}/{}'.format(self.directory, label)
            for img in os.listdir(label_path):
                image_paths.append('{}/{}'.format(label_path, img))
                image_labels.append(self.one_hot(label))
        return image_paths, image_labels

    def shuffle_images(self):
        combined = list(zip(self.image_paths, self.image_labels))
        random.shuffle(combined)  # shuffle, keeping order
        self.image_paths, self.image_labels = zip(*combined)

    def load_image(self, path):
        # img = Image.open(path)
        # arr = np.array(img)
        # return np.asarray(Image.open(path), dtype=np.float32) / 255
        # return imread(path)[..., [2, 1, 0]]
        YUV = False
        img = cv2.imread(path).astype(np.float32) / 255.  # seems to be the fastest
        if YUV:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img

    def one_hot(self, picked_label):
        if not self.use_new_labels:
            one = [0] * len(self.labels)
            one[self.labels.index(picked_label)] = 1
        else:
            one = [0] * len(self.new_labels)
            new_label = self.transform_old_labels[picked_label]
            one[self.new_labels.index(new_label)] = 1

        return one
