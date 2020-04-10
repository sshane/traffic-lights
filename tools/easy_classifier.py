import os
import cv2
try:
    from utils.basedir import BASEDIR
except ImportError:
    BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from threading import Thread
from threading import Lock
import matplotlib.pyplot as plt
import time
try:
    from tensorflow import keras
    has_tf = True
except:
    has_tf = False
import numpy as np
import shutil

os.chdir(BASEDIR)


class EasyClassifier:  # todo: implement smart skip. low skip value when model predicts traffic light, high skip when model predicts none
    def __init__(self):
        global has_tf
        self.data_labels = ['RED', 'GREEN', 'YELLOW', 'NONE']
        self.model_labels = ['SLOW', 'GREEN', 'NONE']
        self.data_dir = 'data'
        self.to_add_dir = '{}/to_add'.format(self.data_dir)
        self.extracted_dir = 'new_data/extracted'
        self.already_classified_dir = 'new_data/extracted/already_classified'
        self.routes = [i for i in os.listdir(self.extracted_dir) if i not in ['already_classified', 'todo', 'temp']]
        self.frame_rate = 20
        self.default_skip = int(0.5 * self.frame_rate)  # amount of frames to skip

        self.skip = 0
        self.user_skip = 0

        self.model_name = 'latest'
        try:
            self.model = keras.models.load_model('models/h5_models/{}.h5'.format(self.model_name))
        except:
            has_tf = False
            self.model = None

        self.max_preloaded_routes = 1  # number of routes to preload (set to 0 if your system locks up or runs out of memory, barely works with 32GB)
        self.show_predictions = False  # set to False if you don't have enough RAM to load all predictions in memory, or you do not want to show predictions
        if not has_tf:
            self.show_predictions = False
        self.preloaded_routes = []
        self.all_routes_done = False
        self.lock = Lock()

        self.make_dirs()

    def start(self):
        Thread(target=self.route_preloader, args=(self.model, )).start()
        self.classifier_manager()

    def classifier_manager(self):
        print('-----\n  Valid inputs:')
        print('  `{class}` - Move image to class folder')
        print('  `skip {num frames}` - Set the number of seconds to skip')
        print('  `skip {num frames} now` - Skip n seconds now')
        print('  `next route` - Skip to next route')
        print('-----')
        while True:
            if len(self.preloaded_routes) > 0:
                this_route = self.preloaded_routes[0]
                with self.lock:
                    del self.preloaded_routes[0]
                self.start_classifying(this_route)
                if len(self.preloaded_routes) > 0:
                    print('Preloaded routes: {}'.format(len(self.preloaded_routes)))
                print('NEXT ROUTE!')
            elif self.all_routes_done and len(self.preloaded_routes) == 0:
                print('All routes classified!')
                return
            else:
                time.sleep(1)

    def route_preloader(self, model):
        for route in self.routes:
            route_dir = '{}/{}'.format(self.extracted_dir, route)
            list_dir = self.sort_list_dir(os.listdir(route_dir))  # sorted using integar values of frame idx
            print('Route: {}'.format(route))
            if self.show_predictions:
                print('Loading all images from route to show predictions, please wait...', flush=True)
                all_imgs = self.load_imgs_from_directory(list_dir, route_dir)
                if len(all_imgs) > 0:
                    print('Loaded all images! Now predicting...', flush=True)
                    with self.lock:
                        self.preloaded_routes.append({'route_predictions': self.predict_multiple(all_imgs, model),
                                                      'route_name': route,
                                                      'route_dir': route_dir,
                                                      'list_dir': list_dir})
                    print('Preloaded route!')
                else:
                    print('Skipping empty folder...')
                    self.move_folder(route_dir, self.already_classified_dir)
                del all_imgs  # free memory
                del list_dir
            else:
                with self.lock:
                    self.preloaded_routes.append({'route_predictions': None,
                                                  'route_name': route,
                                                  'route_dir': route_dir,
                                                  'list_dir': list_dir})

            while len(self.preloaded_routes) >= self.max_preloaded_routes:
                time.sleep(1)  # too many preloaded, wait until user is done with preloaded routes

        self.all_routes_done = True

    def start_classifying(self, route):
        print('Route: {}'.format(route['route_name']))
        for idx, img_name in enumerate(route['list_dir']):
            if self.skip != 0 and self.user_skip == 0:  # this skips ahead in time
                self.skip -= 1
                continue
            else:
                self.skip = int(self.default_skip)

            if self.user_skip != 0:
                self.user_skip -= 1
                continue
            print('At frame: {}'.format(idx))

            img_path = '{}/{}'.format(route['route_dir'], img_name)
            try:
                img = cv2.imread(img_path)
            except:
                print('Skipping corrupted image!')
                continue
            plt.clf()
            plt.imshow(self.crop_image(self.BGR2RGB(img), False))

            if self.show_predictions:
                pred = route['route_predictions'][idx]
                pred_idx = np.argmax(pred)
                plt.title('Prediction: {} ({}%)'.format(self.model_labels[pred_idx], round(pred[pred_idx] * 100, 2)))
            else:
                plt.title('Route: {}, frame: {}'.format(route['route_name'], idx))
            plt.pause(0.01)

            user_out = self.get_true_label()
            if user_out.skip:
                continue
            elif user_out.label is not None:
                correct_label = user_out.label
                print('Moved to {} folder!'.format(correct_label))
                self.move(img_path, '{}/{}/{}'.format(self.to_add_dir, correct_label, img_name))
            elif user_out.next_route:
                print('Skipping to next route!')
                break
            else:
                raise Exception('Unknown command! We shouldn\'t be here...')

        self.reset_skip()
        self.move_folder(route['route_dir'], self.already_classified_dir)

    def sort_list_dir(self, list_dir):  # because the way os and sorted() sorts the files is incorrect but technically correct
        file_nums = [int(file.split('.')[-2]) for file in list_dir]
        return [fi_num for _, fi_num in sorted(zip(file_nums, list_dir))]

    def load_imgs_from_directory(self, list_dir, route_dir):
        imgs = []
        for img_name in list_dir:
            img_path = '{}/{}'.format(route_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:  # skips bad files
                continue
            imgs.append(np.array((img / 255.0), dtype=np.float32))
        return imgs

    def get_true_label(self):
        class ReturnClass:
            label = None
            skip = False
            next_route = False

        return_class = ReturnClass()
        labels = {lbl[0]: lbl for lbl in self.data_labels}

        while True:
            u_input = input('> ').strip(' ').upper()
            if u_input in labels:
                return_class.label = labels[u_input]
                break
            elif u_input in labels.values():
                return_class.label = u_input
                break
            elif 'SKIP' in u_input:
                u_input = u_input.split(' ')
                try:
                    skip_parsed = float(u_input[1])
                    if len(u_input) == 3:  # if followed by 'now', skip now
                        if u_input[-1] == 'NOW':
                            self.user_skip = max(int(skip_parsed * self.frame_rate), 0)
                            self.skip = 0
                            print('Skipping {} frames!'.format(self.user_skip))
                            return_class.skip = True
                            break
                    elif len(u_input) == 2:  # else set default skip
                        self.default_skip = max(int(skip_parsed * self.frame_rate), 0)
                        self.skip = int(self.default_skip)
                        self.user_skip = 0
                        print('Set skipping to {} frames!'.format(self.default_skip))
                        continue
                except:
                    print('Exception when parsing input to skip, try again!')
                    continue
            elif 'NEXT ROUTE' in u_input:
                return_class.next_route = True
                break
            print('Invalid input, try again!')
        return return_class

    def move_folder(self, source, destination):
        shutil.move(source, destination)

    def move(self, source, destination):
        os.rename(source, destination)

    def reset_skip(self):
        self.skip = 0
        self.user_skip = 0

    def predict_multiple(self, imgs, model):
        imgs_cropped = [self.crop_image(img, False) for img in imgs]
        predictions = model.predict(np.array(imgs_cropped))
        return predictions

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
            for lbl in self.data_labels:
                os.makedirs('{}/{}'.format(self.to_add_dir, lbl))
        except:
            pass
        try:
            os.makedirs(self.already_classified_dir)
        except:
            pass


easy_classifier = EasyClassifier()
easy_classifier.start()
