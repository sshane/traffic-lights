import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import shutil
import time

main_dir = 'C:/Git/traffic-lights/data/new_data'
os.chdir(main_dir)

need_class = 'needs_classification'

for file in os.listdir(need_class):
  img_path = os.path.join(need_class, file)

  img = mpimg.imread(os.path.join(need_class, file))

  # plt.imshow(cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR))
  plt.clf()
  t = time.time()
  plt.imshow(img)

  plt.pause(0.01)
  print(file)
  print('Label image: g:green, r:red, y:yellow, n:none')
  print(time.time() -t)
  label = input('Label for image: ').strip().lower()


  if label == 'r':
    t = time.time()
    shutil.move(img_path, os.path.join('RED', file))
    print(time.time() -t)
  elif label == 'g':
    shutil.move(img_path, os.path.join('GREEN', file))
  elif label == 'y':
    shutil.move(img_path, os.path.join('YELLOW', file))
  elif label == 'n':
    shutil.move(img_path, os.path.join('NONE', file))
