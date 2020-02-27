import os
import cv2
import random
from utils.basedir import BASEDIR
import string
from threading import Thread
import matplotlib.pyplot as plt
import time
import keras
import numpy as np
import shutil

os.chdir(BASEDIR)


class EasyClassifier:  # todo: implement smart skip. low skip value when model predicts traffic light, high skip when model predicts none
    def __init__(self):
        self.labels = ['RED', 'GREEN', 'YELLOW', 'NONE']
        self.data_dir = 'data'
        self.to_add_dir = '{}/to_add'.format(self.data_dir)
        self.extracted_dir = 'new_data/extracted'
        self.already_classified_dir = 'new_data/extracted/already_classified'
        self.routes = os.listdir(self.extracted_dir)
        self.frame_rate = 20
        self.default_skip = int(0.5 * self.frame_rate)  # amount of frames to skip

        self.skip = 0
        self.user_skip = 0

        self.model_name = 'high_acc_low_size'
        self.model = keras.models.load_model('models/h5_models/{}.h5'.format(self.model_name))

        self.make_dirs()
        self.show_imgs()

    def show_imgs(self):
        for route in [i for i in self.routes if 'already_classified' not in i]:
            route_dir = '{}/{}'.format(self.extracted_dir, route)
            list_dir = self.sort_list_dir(os.listdir(route_dir))  # sorted using integar values of frame idx
            print('Route: {}'.format(route))
            print('Loading all images from route to predict, please wait...', flush=True)
            all_imgs = self.load_imgs_from_directory(list_dir, route_dir)
            print('Loaded all images! Now predicting...', flush=True)
            predictions = self.predict_multiple(all_imgs)
            del all_imgs  # free unused memory
            print('Valid inputs: [Correct/{class}/skip {num frames}]')
            for idx, img_name in enumerate(list_dir):
                if self.skip != 0 and self.user_skip == 0:  # this skips ahead in time
                    self.skip -= 1
                    continue
                else:
                    self.skip = int(self.default_skip)

                if self.user_skip != 0:
                    self.user_skip -= 1
                    continue
                print('At frame: {}'.format(idx))

                img_path = '{}/{}'.format(route_dir, img_name)

                img = cv2.imread(img_path)
                plt.clf()
                plt.imshow(self.crop_image(self.BGR2RGB(img), False))

                pred = predictions[idx]
                pred_idx = np.argmax(pred)

                plt.title('Prediction: {} ({}%)'.format(self.labels[pred_idx], round(pred[pred_idx] * 100, 2)))
                plt.pause(0.01)

                output_dict = self.get_true_label(self.labels[pred_idx])

                if output_dict['skip']:
                    continue
                elif output_dict['correct']:
                    self.move(img_path, '{}/{}/{}'.format(self.to_add_dir, self.labels[pred_idx], img_name))
                else:
                    correct_label = output_dict['label']
                    self.move(img_path, '{}/{}/{}'.format(self.to_add_dir, correct_label, img_name))

            self.reset_skip()
            self.move_folder(route_dir, self.already_classified_dir)
            print('Next video!')

    def sort_list_dir(self, list_dir):  # because the way os and sorted() sorts the files is incorrect but technically correct
        file_nums = [int(file.split('.')[-2]) for file in list_dir]
        return [fi_num for _, fi_num in sorted(zip(file_nums, list_dir))]

    def load_imgs_from_directory(self, list_dir, route_dir):
        imgs = []
        for img_name in list_dir:
            img_path = '{}/{}'.format(route_dir, img_name)
            img = np.array((cv2.imread(img_path) / 255.0), dtype=np.float32)
            imgs.append(img)
        return imgs

    def get_true_label(self, model_pred):
        correct = ['C', 'YES', 'CORRECT']
        labels = {'R': 'RED', 'G': 'GREEN', 'N': 'NONE', 'Y': 'YELLOW'}

        while True:
            u_input = input('> ').strip(' ').upper()
            if u_input in labels:
                print('Moved to {} folder!'.format(labels[u_input]))
                return {'label': labels[u_input], 'correct': False, 'skip': False}
            elif u_input in labels.values():
                print('Moved to {} folder!'.format(u_input))
                return {'label': u_input, 'correct': False, 'skip': False}
            elif u_input in correct:
                print('Moved to {} folder!'.format(model_pred))
                return {'label': model_pred, 'correct': True, 'skip': False}
            elif 'SKIP' in u_input:
                u_input = u_input.split(' ')
                try:
                    skip_parsed = float(u_input[1])
                    if len(u_input) == 3:  # if followed by 'now', skip now
                        if u_input[-1] == 'NOW':
                            self.user_skip = max(int(skip_parsed * self.frame_rate), 0)
                            self.skip = 0
                            print('Skipping {} frames!'.format(self.user_skip))
                            return {'label': None, 'correct': None, 'skip': True}
                    elif len(u_input) == 2:  # else set default skip
                        self.default_skip = max(int(skip_parsed * self.frame_rate), 0)
                        self.skip = int(self.default_skip)
                        self.user_skip = 0
                        print('Set skipping to {} frames!'.format(self.default_skip))
                        continue
                except:
                    print('Exception when parsing input to skip, try again!')
                    continue
            print('Invalid input, try again!')

    def move_folder(self, source, destination):
        shutil.move(source, destination)

    def move(self, source, destination):
        # print('Moving {} to {}!'.format(source, destination))
        os.rename(source, destination)

    def reset_skip(self):
        self.skip = 0
        self.user_skip = 0

    def predict_multiple(self, imgs):
        imgs_cropped = [self.crop_image(img, True) for img in imgs]
        return self.model.predict(np.array(imgs_cropped))

    def crop_image(self, img_array, crop_top):
        h_crop = 175  # horizontal, 150 is good, need to test higher vals
        if crop_top:
            t_crop = 150  # top, 100 is good. test higher vals
        else:
            t_crop = 0
        y_hood_crop = 665
        return img_array[t_crop:y_hood_crop, h_crop:-h_crop]  # removes 150 pixels from each side, removes hood, and removes 100 pixels from top

    def BGR2RGB(self, arr):
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def make_dirs(self):
        try:
            for lbl in self.labels:
                os.makedirs('{}/{}'.format(self.to_add_dir, lbl))
        except:
            pass
        try:
            os.makedirs(self.already_classified_dir)
        except:
            pass


auto_classifier = EasyClassifier()
