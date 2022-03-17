import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import h5py
import scipy.io as sio

img_path = 'data/cub_raw/images'
seg_path = 'data/cub_raw/segmentations'

target_size = 128

id2path = [l.split() for l in open('data/cub_raw/images.txt')]
id2path = {id: path for id, path in id2path}
split = [l.split() for l in open('data/cub_raw/train_val_test_split.txt')]
split = {id: train_test_idx for id, train_test_idx in split}

train_img = []
train_seg = []
test_img = []
test_seg = []

for id, path in id2path.items():

    img = Image.open(os.path.join(img_path, path)).resize((target_size, target_size), resample=Image.BILINEAR)
    seg = Image.open(os.path.join(seg_path, path[:-4]+'.png')).resize((target_size, target_size), resample=Image.BILINEAR)
    img = np.asarray(img)
    seg = np.asarray(seg) > 255/2

    try:
        img = img.transpose((2, 0, 1)).reshape((3, target_size, target_size))
    except:
        img = np.concatenate([img[None, :, :], img[None, :, :], img[None, :, :]], axis=0)

    try:
        seg = seg.reshape((target_size, target_size))
    except:
        print(seg.shape)
        plt.imshow(img.reshape((3, target_size, target_size)).transpose(1, 2, 0))
        plt.show()
        plt.imshow(np.asarray(seg)[:, :, 0])
        plt.show()
        seg = seg[:, :, 0].reshape((target_size, target_size))

    if split[id] == '0':
        train_img.append(img)
        train_seg.append(seg)

    if split[id] == '2':
        test_img.append(img)
        test_seg.append(seg)

print(len(train_img), len(test_img))

file = h5py.File('data/cub/cub.h5', "w")
file.create_dataset("train_img", np.shape(train_img), h5py.h5t.STD_U8BE, data=train_img)
file.create_dataset("train_seg", np.shape(train_seg), h5py.h5t.STD_U8BE, data=train_seg)
file.create_dataset("test_img", np.shape(test_img), h5py.h5t.STD_U8BE, data=test_img)
file.create_dataset("test_seg", np.shape(test_seg), h5py.h5t.STD_U8BE, data=test_seg)
file.close()
