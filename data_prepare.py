import os
import numpy as np
import cv2
from skimage.transform import resize
import pandas as pd
from skimage.io import imread,imshow

w=128
h=128
c=3

train_img = next(os.walk('stage1_train'))[1]
test_img = next(os.walk('stage1_test'))[1]

X_train = np.zeros((len(train_img),h,w,c),dtype=np.uint8)
Y_train = np.zeros((len(train_img),h,w,1),dtype=np.bool)

X_test = np.zeros((len(test_img),h,w,c),dtype=np.uint8)



for i, name, in enumerate(train_img):
    path = 'stage1_train/' + name
    img = imread(path + '/images/' + name + ".png")[:,:,:c]
    img = resize(img,(h,w), mode='constant', preserve_range=True) 
    X_train[i] = img
    mask = np.zeros((h,w,1), dtype=np.bool)
    for mask_cel in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_cel )
        mask_ = np.expand_dims(resize(mask_,(h,w), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[i] = mask

sizes_test = []
for i, name, in enumerate(test_img):
    path = 'stage1_test/' + name
    img = cv2.imread(path + '/images/' + name + ".png")[:,:,:c]
    sizes_test.append([img.shape[0],img.shape[1]])
    img = resize(img,(h,w), mode='constant', preserve_range=True)
    X_test[i] = img

np.save("X_train.npy", X_train)
#numpy.load("data.npy")

np.save("Y_train.npy", Y_train)

np.save("X_test.npy",X_test)

np.save("sizes_test.npy", sizes_test)

