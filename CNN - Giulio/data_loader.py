import os
import glob
import cv2
import numpy as np
import random
import tensorflow as tf
# from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(data_path = './dataset', norm=True):
    def normalize(data):
        # every image is composed of 96x96 cells containing values from 0 to 255 (RGB)
        # dividing by 255 we get values from 0 to 1, then subtract 0.5 to be close to 0
        data = (data / 255) - 0.5
        return data

    def load_data(mypath):
        # tensor where we put the data, no images now so 0
        # each image will be 96x96 with 3 layers RGB
        data = np.zeros((0, 150, 150, 3), dtype='uint8')
        labels = np.zeros((0,))
        for i, cla in enumerate(mypath):
            filelist = glob.glob(os.path.join(cla, '*')) # path of all images in folder cla
            tmp_data = np.empty((len(filelist), 150, 150, 3), dtype='uint8') # temp array where we store images
            tmp_labels = np.ones((len(filelist),)) * i # temp array where we store labels
            for j, path in enumerate(filelist):
                image = cv2.imread(path) # read each image getting a vector 96x96x3
                image = cv2.resize(image, (150, 150))
                tmp_data[j, :] = image # put it in the previous array
            data = np.concatenate((data, tmp_data))
            labels = np.concatenate((labels, tmp_labels))
        return data, labels

    train_path = glob.glob(os.path.join(data_path, 'train', '*')) # take path of training folders
    train_path.sort() # sort so we are sure that training and test folders are in same order
    print(train_path)
    test_path = glob.glob(os.path.join(data_path, 'test', '*'))
    test_path.sort()
    training_data, training_labels = load_data(train_path) # run function above to get images and labels
    test_data, test_labels = load_data(test_path)
    # perm = np.arange(training_data.shape[0]) # array containing test indices
    # random.shuffle(perm) # randomly shuffle the test data indices
    # test_size = np.floor(training_data.shape[0]*0.3).astype(int)
    # perm = perm[:test_size] # take only the first 1000 (just because previously it was too big)
    # test_data = training_data[perm, :]
    # test_labels = training_labels[perm]
    # training_data = np.delete(training_data, perm, 0)
    # training_labels = np.delete(training_labels, perm, 0)
    if norm:
        training_data = normalize(training_data)
        test_data = normalize(test_data)
    return training_data, training_labels, test_data, test_labels


def augmentation(dataset, labels):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    datagen.fit(dataset)

    aug_data = np.zeros((0, 150, 150, 3), dtype='uint8')
    temp_data = np.empty((9, 150, 150, 3), dtype='uint8')

    for data in dataset:
        data = np.expand_dims(data, axis=0)
        for i, image in enumerate(datagen.flow(data, batch_size=9)):
            temp_data[i, :] = image
            if i + 1 == 9:
                break
        aug_data = np.concatenate((aug_data, data, temp_data))

    aug_labels = np.repeat(labels, 10)

    return aug_data, aug_labels