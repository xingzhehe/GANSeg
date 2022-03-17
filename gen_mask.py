import argparse
import importlib
import json
import math
import os
import numpy as np
import torch
from types import MethodType
from datasets.datasets import get_dataloader
from models.heatmaps import gen_heatmaps
import matplotlib.pyplot as plt
from visualization import label2rgb, get_pt_color
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--segmenter_log', type=str, default='log/seg_celeba_wild_k8')
parser.add_argument('--test_class_name', type=str, default='mafl_wild_test')
parser.add_argument('--data_root', type=str, default='data/celeba_wild')
parser.add_argument('--save_root', type=str, default='saved_mask/celeba_wild_k8')
parser.add_argument('--checkpoint', type=int, default=10)
args = parser.parse_args()

with open(os.path.join(args.segmenter_log, 'parameters.json'), 'rt') as f:
    segmenter_log = json.load(f)

log_split = args.segmenter_log.split('_')
for dummy in log_split:
    if dummy[0] == 'k':
        n_keypoints = int(dummy[1:])

kp_color, pt_color = get_pt_color(n_keypoints, 4)

device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

dataset = get_dataloader(data_root=args.data_root, batch_size=200, class_name=args.test_class_name, drop_last=False,
                         image_size=128)
segmenter = importlib.import_module('models.' + segmenter_log['model'])
segmenter = segmenter.Segmenter({'n_classes': n_keypoints + 1}).to(device)

seg_checkpoint = torch.load(os.path.join(args.segmenter_log, 'checkpoints', 'epoch_{}.model'.format(args.checkpoint)),
                            map_location=lambda storage, location: storage)
segmenter.load_state_dict(seg_checkpoint['detector'])
segmenter.to(device)
segmenter.eval()
torch.autograd.set_grad_enabled(False)

os.makedirs(args.save_root, exist_ok=True)

sample = next(iter(dataset))
for i in range(200):
    img = sample['img'][i]
    mask = segmenter({'img': img.to(device).unsqueeze(0)})['seg'].max(dim=1)[1].cpu()[0]
    img = img.numpy().transpose((1, 2, 0)) * 0.5 + 0.5
    masked_img = label2rgb(mask.numpy(), img, segment_color=kp_color, alpha=0.5)
    fg_mask = (mask != 0).int().unsqueeze(-1).numpy()
    masked_img = masked_img * fg_mask + img * (1 - fg_mask)
    masked_img = np.uint8(masked_img*255)
    Image.fromarray(masked_img).save(os.path.join(args.save_root, '{}.png'.format(i)))
