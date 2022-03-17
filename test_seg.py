import argparse
import importlib
import json
import os
import torch
import torch.utils.data
from torchvision import transforms
from sklearn import linear_model
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets.datasets import get_dataloader
from models.heatmaps import gen_grid2d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from torch import nn
from visualization import label2rgb, get_pt_color
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--segmenter_log', type=str, default='log/seg_celeba_wild_k8')
parser.add_argument('--test_class_name', type=str, default='mafl_wild_test')
parser.add_argument('--data_root', type=str, default='data/celeba_wild')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--checkpoint', type=int, default=10)
args = parser.parse_args()

with open(os.path.join(args.segmenter_log, 'parameters.json'), 'rt') as f:
    segmenter_log = json.load(f)

log_split = args.segmenter_log.split('_')
for dummy in log_split:
    if dummy[0] == 'k':
        n_keypoints = int(dummy[1:])

device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
seg_model = importlib.import_module('models.' + segmenter_log['model'])
segmenter = seg_model.Segmenter({'n_classes': n_keypoints + 1}).to(device)
seg_checkpoint = torch.load(os.path.join(segmenter_log['log'], 'checkpoints', 'epoch_{}.model'.format(args.checkpoint)),
                            map_location=lambda storage, location: storage)
segmenter.load_state_dict(seg_checkpoint['detector'])
segmenter.eval()
del seg_checkpoint

test_dataloader = get_dataloader(data_root=args.data_root, class_name=args.test_class_name,
                                 batch_size=segmenter_log['batch_size'],
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False, image_size=segmenter_log['image_size'])
if 'mafl' in args.test_class_name:
    train_dataloader = get_dataloader(data_root=args.data_root, class_name=args.test_class_name[:-5],
                                      batch_size=segmenter_log['batch_size'],
                                      num_workers=args.num_workers, pin_memory=True, drop_last=False, image_size=segmenter_log['image_size'])
elif 'taichi' in args.test_class_name:
    train_dataloader = get_dataloader(data_root=args.data_root, class_name='taichi_reg_train',
                                      batch_size=segmenter_log['batch_size'],
                                      num_workers=args.num_workers, pin_memory=True, drop_last=False, image_size=segmenter_log['image_size'])


def mask2keypoints(kp_mask):
    grid_size = kp_mask.shape[-1]
    grid = gen_grid2d(grid_size, device=kp_mask.device, left_end=0, right_end=segmenter_log['image_size'])
    grid = grid.reshape(1, 1, grid_size, grid_size, 2)
    kp_mask = kp_mask.reshape(kp_mask.shape[0], n_keypoints, grid_size, grid_size, 1)
    normalized_kp_mask = kp_mask / (1e-6 + kp_mask.sum(dim=(2, 3), keepdim=True))
    return (grid * normalized_kp_mask).sum(dim=(2, 3))


def evaluate_detection():
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    with torch.no_grad():
        for batch_index, real_batch in enumerate(train_dataloader):
            real_batch['img'] = real_batch['img'].to(device)
            masks = segmenter(real_batch)['seg']
            masks = F.softmax(masks, dim=1)[:, 1:, :, :]
            train_X.append(mask2keypoints(masks).detach().cpu())
            train_y.append(real_batch['keypoints'])
        train_X = torch.cat(train_X)
        train_y = torch.cat(train_y)

        for batch_index, real_batch in enumerate(test_dataloader):
            real_batch['img'] = real_batch['img'].to(device)
            masks = segmenter(real_batch)['seg']
            masks = F.softmax(masks, dim=1)[:, 1:, :, :]
            test_X.append(mask2keypoints(masks).detach().cpu())
            test_y.append(real_batch['keypoints'])
        test_X = torch.cat(test_X)
        test_y = torch.cat(test_y)

    if 'mafl' in args.test_class_name:
        with torch.no_grad():
            train_X = train_X.reshape(train_X.shape[0], -1)
            train_y = train_y.reshape(train_y.shape[0], -1)
            test_X = test_X.reshape(test_X.shape[0], -1)
            test_y = test_y.reshape(test_y.shape[0], -1)
            try:
                beta = (train_X.T @ train_X).inverse() @ train_X.T @ train_y
            except:
                beta = (train_X.T @ train_X + torch.eye(train_X.shape[-1])).inverse() @ train_X.T @ train_y
            pred_y = test_X @ beta
            unnormalized_loss = (pred_y - test_y).reshape(test_X.shape[0], 5, 2).norm(dim=-1)
            eye_distance = (test_y.reshape(test_y.shape[0], 5, 2)[:, 0, :] - test_y.reshape(test_y.shape[0], 5, 2)[:, 1, :]).norm(dim=-1)
            normalized_loss = (unnormalized_loss / eye_distance.unsqueeze(1)).mean()
        return normalized_loss.item()

    elif 'taichi' in args.test_class_name:
        scores = []
        num_gnd_kp = 18
        for i in range(num_gnd_kp):
            for j in range(2):
                index = (train_y[:, i, j] + 1).abs() > 1e-6
                features = train_X[index]
                features = features.reshape(features.shape[0], -1)
                label = train_y[index, i, j]
                features = torch.cat([features, torch.ones_like(features[:, -1:])], dim=1)
                try:
                    beta = (features.T @ features).inverse() @ features.T @ label
                except:
                    beta = (features.T @ features + torch.eye(features.shape[-1])).inverse() @ features.T @ label

                index_test = (test_y[:, i, j] + 1).abs() > 1e-6
                features = test_X[index_test]
                features = features.reshape(features.shape[0], -1)
                features = torch.cat([features, torch.ones_like(features[:, -1:])], dim=1)
                label = test_y[index_test, i, j]
                pred_label = features @ beta
                score = (pred_label - label).abs().mean()
                scores.append(score.item())
        return np.sum(scores)


def cal_iou(outputs, labels):
    # BATCH x H x W shape
    assert outputs.shape == labels.shape

    intersection = (outputs & labels).float().sum((-1, -2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((-1, -2))  # Will be zzero if both are 0

    iou = intersection / (union + 1e-6)  # We smooth our devision to avoid 0/0

    return iou


def evaluate_iou():
    total_iou = 0
    with torch.no_grad():
        for batch_index, real_batch in enumerate(test_dataloader):
            real_batch['img'] = real_batch['img'].to(device)
            pred_seg = segmenter(real_batch)['seg'].max(dim=1)[1]
            fg_mask = (pred_seg != 0).int()
            total_iou += cal_iou(fg_mask, real_batch['seg'].to(device)).sum().item()
        iou = total_iou / len(test_dataloader.dataset)
    return iou


if __name__ == '__main__':
    with open(os.path.join(args.segmenter_log, 'parameters.json'), 'r') as f:
        para = json.load(f)

    if 'mafl' in args.test_class_name or 'taichi' in args.test_class_name:
        test_kp_error = evaluate_detection()
        print('test_kp_error', test_kp_error)
        para['test_kp_error'] = test_kp_error

    if 'cub' in args.test_class_name or 'flower' in args.test_class_name or 'taichi' in args.test_class_name:
        test_iou = evaluate_iou()
        print('test_iou', test_iou)
        para['test_iou'] = test_iou

    with open(os.path.join(args.segmenter_log, 'parameters.json'), 'w') as f:
        json.dump(para, f, indent=2)