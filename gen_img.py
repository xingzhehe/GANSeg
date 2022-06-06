import argparse
import importlib
import json
import os
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.heatmaps import gen_heatmaps
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--generator_log', type=str, default='log/gan_celeba_wild_k8')
parser.add_argument('--save_root', type=str, default='saved_image/celeba_wild_k8')
parser.add_argument('--checkpoint', type=int, default=30)
args = parser.parse_args()

with open(os.path.join(args.generator_log, 'parameters.json'), 'rt') as f:
    generator_args = json.load(f)

device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
gen_model = importlib.import_module('models.' + generator_args['model'])
generator = gen_model.Generator({'z_dim': generator_args['z_dim'], 'n_keypoints': generator_args['n_keypoints'], 'n_per_kp': generator_args['n_per_kp'],
                                 'n_embedding': generator_args['n_embedding'], 'image_size': generator_args['image_size'],
                                 'single_final': generator_args['single_final'], 'use_linear': generator_args['use_linear'],
                                 'smaller_init_mask': generator_args['smaller_init_mask'],
                                 'feature_map_sizes': generator_args['feature_map_sizes'],
                                 'feature_map_channels': generator_args['feature_map_channels'],
                                 'class_name': generator_args['class_name'],}).to(device)
gen_checkpoint = torch.load(os.path.join(generator_args['log'], 'checkpoints', 'epoch_{}.model'.format(args.checkpoint)),
                            map_location=lambda storage, location: storage)
generator.load_state_dict(gen_checkpoint['generator'])
torch.autograd.set_grad_enabled(False)
del gen_checkpoint


if __name__ == '__main__':
    os.makedirs(args.save_root, exist_ok=True)
    generator.eval()
    index = 0
    with torch.no_grad():
        while index < 10:
            input_batch = {'input_noise{}'.format(noise_i): torch.randn(64, *noise_shape).to(device)
                           for noise_i, noise_shape in enumerate(generator.noise_shapes)}
            input_batch['bg_trans'] = torch.rand(64, 1, 2).to(device) * 2 - 1
            imgs = generator(input_batch)['img'].detach().cpu().permute(0, 2, 3, 1) * 0.5 + 0.5
            imgs = torch.clamp(imgs, 0, 1)
            for img in imgs:
                img = np.uint8(img * 255)
                Image.fromarray(img).save(os.path.join(args.save_root, '{}.png'.format(index)))
                index += 1
