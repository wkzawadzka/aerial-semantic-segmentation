import h5py
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from  tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
import cv2
import random

class DataLoader:
    def __init__(self, augmentation = True):
        # load files
        self.train_data = h5py.File('./data/tree_train.h5/tree_train.h5', 'r')
        test_data = h5py.File('./data/tree_test.h5/tree_test.h5', 'r')

        # extract training and testing data from files
        self.X_train, self.y_train, self.X_test = np.array(self.train_data['X']), np.array(self.train_data['y']), np.array(test_data['X'])

        # perform data augmentation if augmentation parameter is True
        if augmentation:
            self.X_train, self.y_train = self.perform_augmentation(800)

        # preprocess images (normalization)
        self.X_train = np.array(list(map(self.preprocess, self.X_train)))
        self.X_test = np.array(list(map(self.preprocess, self.X_test)))

        # split the data into training and validation with a split ratio of 0.8
        self.X_valid, self.y_valid = self.split_data()


    def preprocess(self, img):
        ''' preprocess single image '''
        img = img/255. # normalization to range [0,1]
        return img

    def split_data(self, split_size: float = 0.8):
        ''' split the data into training and validation with the deafult split of 0.8 '''
        # get index for splitting
        split_id = int(split_size*self.X_train.shape[0])
        split_id_y = int(split_size*self.X_train.shape[0])
        # divide into validation and training datasets
        X_valid, y_valid = self.X_train[split_id:], self.y_train[split_id_y:]
        self.X_train, self.y_train = self.X_train[:split_id], self.y_train[:split_id_y]

        return (X_valid, y_valid)

    def perform_augmentation(self, size:int):
        ''' expand dataset by @size new augumented examples '''
        # access dataset
        X_train = self.X_train # images
        y_train = self.y_train # masks

        # prepare transorm procedure
        transform = A.Compose([
                A.RandomCrop(width=128, height=128), # 128x128 image
                A.HorizontalFlip(p=0.5), # flip horizontally with probability 0.5
                A.RandomBrightnessContrast(p=0.2), # change brightness with probability 0.2
                A.RandomRotate90(p=1), # random rotate with probability 1
            ])
        
        # get set of random data examples
        random_ids = random.sample(range(0, self.X_train.shape[0]), size)

        # perform data augumentation on given examples
        for id in random_ids:
            # transorm image and mask at the same time (same transformation)
            transformed = transform(image=self.X_train[id], mask=self.y_train[id])
            X_train = np.insert(X_train, 0, transformed['image'], axis=0) # insert to images dataset
            y_train = np.insert(y_train, 0, transformed['mask'], axis=0) # insert to masks dataset

        # return expanded datasets
        return (X_train, y_train)


