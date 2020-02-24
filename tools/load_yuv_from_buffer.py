import numpy as np
import cv2
import matplotlib.pyplot as plt

with open('C:/Git/buffer', 'rb') as f:
    buf = [i for i in f.read()]

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 3 / 2)
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)
        print(self.shape)

    def read_raw(self):
        print(self.frame_len)
        raw = self.f.read(self.frame_len)
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape(self.shape)

        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV420p2RGB)
        return ret, bgr



#filename = "data/20171214180916RGB.yuv"
filename = "C:/Git/buffer"
size = (874, 1164)
cap = VideoCaptureYUV(filename, size)


ret, frame = cap.read()
if ret:
    cv2.imshow("frame", frame)
