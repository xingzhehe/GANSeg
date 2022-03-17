import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import h5py
import scipy.io as sio

img_path = 'data/flower_raw/jpg'
seg_path = 'data/flower_raw/segmim'

target_size = 128

split = sio.loadmat('data/flower_raw/setid.mat')

print(split.keys())
print(len(split['trnid'][0]), len(split['valid'][0]), len(split['tstid'][0]))
print(split['tstid'])

train_img = []
train_seg = []
test_img = []
test_seg = []

for idx in split['tstid'][0]:
    img_file_name = 'image_{}.jpg'.format(str(idx).zfill(5))
    seg_file_name = 'segmim_{}.jpg'.format(str(idx).zfill(5))

    img = Image.open(os.path.join(img_path, img_file_name)).resize((target_size, target_size), resample=Image.BILINEAR)
    seg = Image.open(os.path.join(seg_path, seg_file_name)).resize((target_size, target_size), resample=Image.BILINEAR)

    img = np.asarray(img)
    seg = np.asarray(seg)
    seg = 1 - ((seg[:, :, 0:1] == 0) + (seg[:, :, 1:2] == 0) + (seg[:, :, 2:3] == 254))

    img = img.transpose((2, 0, 1)).reshape((3, target_size, target_size))
    train_img.append(img)
    train_seg.append(seg.squeeze(-1))


for idx in split['trnid'][0]:
    img_file_name = 'image_{}.jpg'.format(str(idx).zfill(5))
    seg_file_name = 'segmim_{}.jpg'.format(str(idx).zfill(5))

    img = Image.open(os.path.join(img_path, img_file_name)).resize((target_size, target_size), resample=Image.BILINEAR)
    seg = Image.open(os.path.join(seg_path, seg_file_name)).resize((target_size, target_size), resample=Image.BILINEAR)

    img = np.asarray(img)
    seg = np.asarray(seg)
    seg = 1 - ((seg[:, :, 0:1] == 0) + (seg[:, :, 1:2] == 0) + (seg[:, :, 2:3] == 254))

    img = img.transpose((2, 0, 1)).reshape((3, target_size, target_size))
    test_img.append(img)
    test_seg.append(seg.squeeze(-1))

print(len(train_img), len(test_img))

file = h5py.File('data/flower/flower.h5', "w")
file.create_dataset("train_img", np.shape(train_img), h5py.h5t.STD_U8BE, data=train_img)
file.create_dataset("train_seg", np.shape(train_seg), h5py.h5t.STD_U8BE, data=train_seg)
file.create_dataset("test_img", np.shape(test_img), h5py.h5t.STD_U8BE, data=test_img)
file.create_dataset("test_seg", np.shape(test_seg), h5py.h5t.STD_U8BE, data=test_seg)
file.close()
