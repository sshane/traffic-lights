import os
import cv2
import random
from utils.basedir import BASEDIR
import string
from threading import Thread
import time

BASEDIR = os.path.join(BASEDIR, 'new_data')
extracted_dir = '{}/extracted'.format(BASEDIR)
os.chdir(BASEDIR)

num_threads = 0
max_threads = 10
valid_extension = '.hevc'
working_file = '{}/delete_to_stop'.format(BASEDIR)  # delete this file to stop extraction process after current video is done


def write_frame(path, img, ret):
    global num_threads
    num_threads += 1
    if ret:
        cv2.imwrite(path, img)
    num_threads -= 1


def mk_dirs(save_dir, ignore_existing=False):
    if not os.path.exists(extracted_dir):
        os.mkdir(extracted_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif not ignore_existing:
        raise Exception('Video already extracted: {}'.format(save_dir))


def create_working_file():
    with open(working_file, 'w') as f:
        f.write('')


def stop_working():
    return not os.path.exists(working_file)

def wait_for_threads():
    global num_threads
    while num_threads >= max_threads:
        time.sleep(1 / max_threads)


def rm_video(path):
    os.remove(path)


def extract_frames():
    global num_threads
    create_working_file()
    for video in os.listdir('{}/videos'.format(BASEDIR)):
        if video[-5:] != valid_extension:  # not video file, skip
            continue
        video_path = '{}/videos/{}'.format(BASEDIR, video)
        save_name = '--'.join(video.split('--')[:3])
        save_dir = '{}/{}'.format(extracted_dir, save_name)
        mk_dirs(save_dir, ignore_existing=False)

        cap = cv2.VideoCapture(video_path)
        ret = True
        idx = 0
        print('Extracting: {}'.format(video))
        while ret:
            ret, frame = cap.read()
            frame_path = '{}/{}.{}.png'.format(save_dir, save_name, idx)
            t = Thread(target=write_frame, args=(frame_path, frame, ret))
            t.start()
            wait_for_threads()
            idx += 1
        cap.release()
        cv2.destroyAllWindows()
        wait_for_threads()  # wait for all threads to finish before starting next video
        rm_video(video_path)  # delete video once done
        if stop_working():
            print('Working file deleted, stopping!')
            return
    print('\nFinished extracting all videos!')



extract_frames()
