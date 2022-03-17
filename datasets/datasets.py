import torch
from torchvision import datasets, transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
import torchvision
import h5py
import pandas


class CelebAWildTrain(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'celeba_wild.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])
            self.keypoints = torch.from_numpy(hf['train_landmark'][...]) * image_size / 128
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx].float() / 255),
                  'keypoints': self.keypoints[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class MAFLWildTrain(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'celeba_wild.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['mafl_train_img'][...])
            self.keypoints = torch.from_numpy(hf['mafl_train_landmark'][...]) * image_size / 128
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx].float() / 255),
                  'keypoints': self.keypoints[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class MAFLWildTest(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'celeba_wild.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['mafl_test_img'][...])
            self.keypoints = torch.from_numpy(hf['mafl_test_landmark'][...]) * image_size / 128
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx].float() / 255),
                  'keypoints': self.keypoints[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TaichiTrain(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TaichiRegressionTrain(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.imgs = []
        self.poses = []

        with open(os.path.join(data_root, 'landmark', 'taichi_train_gt.pkl'), 'rb') as f:
            pose_file = pandas.read_pickle(f)

        for i in range(len(pose_file)):
            image_file = pose_file.file_name[i]
            img = Image.open(os.path.join(data_root, 'eval_images', 'taichi-256', 'train', image_file))
            img = img.resize((image_size, image_size), resample=Image.BILINEAR)
            self.imgs.append(np.asarray(img) / 255)
            self.poses.append(pose_file.value[i])

        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torch.tensor(self.imgs).float().permute(0, 3, 1, 2)
        for i in range(len(self.imgs)):
            self.imgs[i] = self.transform(self.imgs[i])
        self.imgs = self.imgs.contiguous()
        self.poses = torch.tensor(self.poses).float()
        self.poses = torch.cat([self.poses[:, :, 1:2], self.poses[:, :, 0:1]], dim=2)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx], 'keypoints': self.poses[idx]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TaichiTest(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()

        self.data_root = data_root
        self.imgs = []
        self.segs = []
        self.poses = []

        with open(os.path.join(data_root, 'landmark', 'taichi_test_gt.pkl'), 'rb') as f:
            pose_file = pandas.read_pickle(f)

        for i in range(len(pose_file)):
            image_file = pose_file.file_name[i]
            img = Image.open(os.path.join(data_root, 'eval_images', 'taichi-256', 'test', image_file))
            img = img.resize((image_size, image_size), resample=Image.BILINEAR)
            seg = Image.open(os.path.join(data_root, 'taichi-test-masks', image_file))
            seg = seg.resize((image_size, image_size), resample=Image.BILINEAR)
            self.imgs.append(np.asarray(img) / 255)
            self.segs.append(np.asarray(seg) / 255)
            self.poses.append(pose_file.value[i])

        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torch.tensor(self.imgs).float().permute(0, 3, 1, 2)
        for i in range(len(self.imgs)):
            self.imgs[i] = self.transform(self.imgs[i])
        self.imgs = self.imgs.contiguous()
        self.segs = torch.tensor(self.segs).int()
        self.poses = torch.tensor(self.poses).float()
        self.poses = torch.cat([self.poses[:, :, 1:2], self.poses[:, :, 0:1]], dim=2)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx], 'seg': self.segs[idx], 'keypoints': self.poses[idx]}
        return sample

    def __len__(self):
        return len(self.imgs)


class Flower(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        data_file = 'flower.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])

        self.imgs = torch.cat((self.imgs, self.imgs, self.imgs), dim=0)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Resize(image_size),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        return {'img': self.transform(self.imgs[index].float() / 255)}

    def __len__(self):
        return len(self.imgs)


class FlowerTest(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        data_file = 'flower.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['test_img'][...])
            self.segs = torch.from_numpy(hf['test_seg'][...])

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        return {'img': self.transform(self.imgs[index].float() / 255), 'seg': self.segs[index]}

    def __len__(self):
        return len(self.imgs)


class CUB(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        data_file = 'cub.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])

        self.imgs = torch.cat((self.imgs, self.imgs, self.imgs), dim=0)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Resize(image_size),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        return {'img': self.transform(self.imgs[index].float() / 255)}

    def __len__(self):
        return len(self.imgs)


class CUBTest(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        data_file = 'cub.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['test_img'][...])
            self.segs = torch.from_numpy(hf['test_seg'][...])

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        return {'img': self.transform(self.imgs[index].float() / 255), 'seg': self.segs[index]}

    def __len__(self):
        return len(self.imgs)


class Custom(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        self.file_paths = [os.path.join(data_root, file_name) for file_name in os.listdir(data_root)
                           if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        img = Image.open(self.file_paths[index])
        return {'img': self.transform(img)}

    def __len__(self):
        return len(self.file_paths)


def get_dataset(data_root, image_size, class_name='all'):
    if class_name == 'flower':
        return Flower(data_root, image_size)
    elif class_name == 'flower_test':
        return FlowerTest(data_root, image_size)
    elif class_name == 'taichi_reg_train':
        return TaichiRegressionTrain(data_root, image_size)
    elif class_name == 'taichi_test':
        return TaichiTest(data_root, image_size)
    elif class_name == 'cub':
        return CUB(data_root, image_size)
    elif class_name == 'cub_test':
        return CUBTest(data_root, image_size)
    elif class_name == 'celeba_wild':
        return CelebAWildTrain(data_root, image_size)
    elif class_name == 'taichi':
        return TaichiTrain(data_root, image_size)
    elif class_name == 'mafl_wild':
        return MAFLWildTrain(data_root, image_size)
    elif class_name == 'mafl_wild_test':
        return MAFLWildTest(data_root, image_size)
    elif class_name == 'custom':
        return Custom(data_root, image_size)
    else:
        raise ValueError


def get_dataloader(data_root, class_name, image_size, batch_size, num_workers=0, pin_memory=True, drop_last=True):
    dataset = get_dataset(data_root=data_root, image_size=image_size, class_name=class_name)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
