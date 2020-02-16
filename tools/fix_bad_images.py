import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.chdir('C:/Git/heyo-new-stuff-for-ya')
for img_file in os.listdir('.'):
    img = cv2.imread(img_file)
    if img is not None:
        os.rename(img_file, '../good_stuff/{}'.format(img_file))