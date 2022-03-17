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
parser.add_argument('--model', type=str, default='segmenter')
parser.add_argument('--generator_log', type=str, default='log/gan_celeba_wild_k8')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--data_root', type=str, default='data/cub')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--checkpoint', type=int, default=30)
args = parser.parse_args()

with open(os.path.join(args.generator_log, 'parameters.json'), 'rt') as f:
    generator_args = json.load(f)

args.log = 'seg_{0}_k{1}'.format(generator_args['class_name'], generator_args['n_keypoints'])
args.log = os.path.join('log', args.log)

args.image_size = generator_args['image_size']

os.makedirs(args.log, exist_ok=True)
with open(os.path.join(args.log, 'parameters.json'), 'wt') as f:
    json.dump(args.__dict__, f, indent=2)

device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
gen_model = importlib.import_module('models.' + generator_args['model'])
generator = gen_model.Generator(generator_args).to(device)
gen_checkpoint = torch.load(os.path.join(generator_args['log'], 'checkpoints', 'epoch_{}.model'.format(args.checkpoint)),
                            map_location=lambda storage, location: storage)
generator.load_state_dict(gen_checkpoint['generator'])
del gen_checkpoint
det_model = importlib.import_module('models.' + args.model)
segmenter = det_model.Segmenter({'n_classes': generator_args['n_keypoints'] + 1}).to(device)
optim = torch.optim.Adam(segmenter.parameters(), lr=args.lr)

generator = torch.nn.DataParallel(generator)


def cal_dice_loss(pred, target, ep=1e-8):
    intersection = 2 * torch.sum(pred * target, dim=(-1, -2)) + ep
    union = torch.sum(pred, dim=(-1, -2)) + torch.sum(target, dim=(-1, -2)) + ep
    loss = 1 - intersection / union
    return loss.mean()


def cal_loss(pred, target, ce_weight=0.5):
    ce = F.cross_entropy(pred, target)

    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=generator_args['n_keypoints'] + 1).permute(0, 3, 1, 2).float()
    dice = cal_dice_loss(pred, target)

    loss = ce * ce_weight + dice * (1 - ce_weight)

    return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.imgs = []
        self.segs = []
        generator.module.eval()
        with torch.no_grad():
            for _ in range(1000):
                input_batch = {'input_noise{}'.format(noise_i): torch.randn(args.batch_size, *noise_shape).to(device)
                               for noise_i, noise_shape in enumerate(generator.module.noise_shapes)}
                input_batch['bg_trans'] = torch.rand(args.batch_size, 1, 2).to(device) * 2 - 1
                output_batch = generator(input_batch)
                self.imgs.append(output_batch['img'].cpu())
                self.segs.append(torch.cat([output_batch['bg_mask'], output_batch['kp_mask']], dim=1).max(dim=1)[1].cpu())
        self.imgs = torch.cat(self.imgs)
        self.segs = torch.cat(self.segs)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx], 'seg': self.segs[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


def train_one_epoch():
    generator.eval()
    segmenter.train()
    total_loss = 0
    generated_dataloader = torch.utils.data.DataLoader(Dataset(),
              batch_size=args.batch_size, shuffle=True,
              num_workers=generator_args['num_workers'], pin_memory=True, drop_last=True)

    for batch_index, batch in enumerate(generated_dataloader):
        optim.zero_grad()
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = cal_loss(segmenter(batch)['seg'], batch['seg'])
        loss.backward()
        total_loss += loss.item()
        optim.step()

    return total_loss / len(generated_dataloader.dataset)


if __name__ == '__main__':
    writer = SummaryWriter(os.path.join(args.log, 'runs'))
    checkpoint_dir = os.path.join(args.log, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(10):
        train_loss = train_one_epoch()

        torch.save(
            {
                'detector': segmenter.state_dict(),
                'optim': optim.state_dict(),
            },
            os.path.join(checkpoint_dir, 'epoch_{}.model'.format(epoch))
        )
