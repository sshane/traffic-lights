import numpy as np
import cv2
import matplotlib.pyplot as plt

with open('S:/Git/cropped', 'r') as f:
    buf = [float(i) for i in f.read().split('\n') if i != '']
    # buf = [i for i in f.read()]

buf = np.array([i for i in buf])
buf = buf.reshape((515, 814, 3)).astype(np.float32)

buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)

plt.imshow(buf)
plt.show()
