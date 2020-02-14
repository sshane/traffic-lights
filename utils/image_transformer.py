import cv2
import numpy as np
import time
import os
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import random
import scipy


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    t = time.time()
    x = np.rollaxis(x, channel_axis, 0)
    print(time.time() - t)
    final_affine_matrix = transform_matrix[:2, :2]
    print(time.time() - t)
    final_offset = transform_matrix[:2, 2]
    print(time.time() - t)
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    print(time.time() - t)
    x = np.stack(channel_images, axis=0)
    print(time.time() - t)
    x = np.rollaxis(x, 0, channel_axis + 1)
    print(time.time() - t)
    return x


class ImageTransformer:
    def __init__(self, input_size=(), crop_size=(), num_output_imgs=0, rotation_range=0, zoom_range=[], limit_samples=0, fill_mode='nearest', cval=0.):
        self.input_size = input_size
        self.crop_size = crop_size
        self.num_output_imgs = num_output_imgs
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.limit_samples = limit_samples
        self.fill_mode = fill_mode
        self.cval = cval

        self.img_channel_axis = 3 - 1
        self.original_images = []

    def load_images(self, directory, cls):
        loaded_images = []
        for idx, img in enumerate(os.listdir(directory)):
            try:
                if idx >= self.limit_samples:
                    break
                img = cv2.imread('{}/{}'.format(directory, img))
                self.original_images.append(img)
                for _ in range(self.num_output_imgs):
                    loaded_images.append(img)
            except:
                pass
        print('{}: Loaded {} images!'.format(cls, len(self.original_images)))
        return np.array(loaded_images)

    def transform_from_directory(self, directory):
        cls = directory.split('/')[-1]
        print(cls)
        images = self.load_images(directory, cls)
        return images
        print('{}: Rotating!'.format(cls), flush=True)
        images = self.rotate_images(images)
        print('{}: Zooming!'.format(cls), flush=True)
        images = self.zoom_images(images)
        print('{}: Flipping!'.format(cls), flush=True)
        images = self.flip_images(images)
        print('{}: Cropping!'.format(cls), flush=True)
        images = self.crop_images(images)
        return images

    def rotate_images(self, images):
        rotated_images = []
        for idx, image in enumerate(images):
            # print(idx, flush=True)

            theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))

            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                       [np.sin(theta), np.cos(theta), 0],
                                       [0, 0, 1]])

            h, w = self.input_size
            rotation_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

            image = apply_transform(image, rotation_matrix, self.img_channel_axis,
                                    fill_mode=self.fill_mode, cval=self.cval)
            print('--------\n')
            rotated_images.append(image)
        return rotated_images

    def flip_images(self, images):
        flipped_images = []
        for image in images:
            if random.choice(['flip', 'no don\'t']) == 'flip':
                flipped_images.append(np.fliplr(image))
            else:
                flipped_images.append(image)
        return flipped_images

    def zoom_images(self, images):
        zoomed_images = []
        for image in images:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            h, w = self.input_size
            transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
            image = apply_transform(image, transform_matrix, self.img_channel_axis,
                                    fill_mode=self.fill_mode, cval=self.cval)
            zoomed_images.append(image)
        return zoomed_images

    def crop_images(self, images):
        cropped_images = [img[0:self.crop_size[1], 0:self.crop_size[0]] for img in images]
        for image in self.original_images:
            cropped_images.append(image[0:self.crop_size[1], 0:self.crop_size[0]])  # crop and add original images
        return cropped_images


img_transformer = ImageTransformer(input_size=(1164, 874), crop_size=(1164, 665), num_output_imgs=10, limit_samples=1000, zoom_range=[0.88, 1.12], rotation_range=3.5)
imgs = img_transformer.transform_from_directory('C:/Git/traffic-lights/data/YELLOW')
# zoomed = img_transformer.zoom_images(imgs)
