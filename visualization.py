import argparse
import importlib
import json
import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets.datasets import get_dataloader
import numpy as np
from matplotlib import colors


def get_pt_color(n_keypoints, n_per_kp):
    if n_keypoints == 4:
        colormap = ('red', 'blue', 'yellow', 'green')
    else:

        colormap = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                    'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                    'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                    'honeydew', 'thistle')[:n_keypoints]
    pt_color = []
    kp_color = []
    for i in range(n_keypoints):
        for _ in range(n_per_kp):
            pt_color.append(colors.to_rgb(colormap[i]))
        kp_color.append(colors.to_rgb(colormap[i]))
    pt_color = np.array(pt_color)
    kp_color = np.array(kp_color)

    return kp_color, pt_color


def label2rgb(mask, image, segment_color, alpha=0.7):
    color_mask = np.zeros_like(image)
    img_size = image.shape[0]

    for i in range(segment_color.shape[0] + 1):
        if i == 0:  # bg_label
            color = np.array((0, 0, 0))
        else:
            color = segment_color[i-1]
        color_mask += (mask == i).reshape((img_size, img_size, 1)) * color.reshape((1, 1, 3))

    return (1-alpha) * image + alpha * color_mask


if __name__ == '__main__':
    kp_color, pt_color = get_pt_color(32, 4)
