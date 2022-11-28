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
from visualization import label2rgb, get_pt_color


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='generator')
parser.add_argument('--dis_model', type=str, default='discriminator')
parser.add_argument('--z_dim', type=int, default=256)
parser.add_argument('--n_keypoints', type=int, default=8)
parser.add_argument('--n_per_kp', type=int, default=4)
parser.add_argument('--feature_map_sizes', type=str, default='32,32,64,128')
parser.add_argument('--feature_map_channels', type=str, default='512,256,128,64')
parser.add_argument('--single_final', type=int, default=0)
parser.add_argument('--use_linear', type=int, default=0)
parser.add_argument('--smaller_init_mask', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr_gen', type=float, default=1e-4)
parser.add_argument('--lr_disc', type=float, default=4e-4)
parser.add_argument('--con_penalty_coef', type=float, default=10)
parser.add_argument('--area_penalty_coef', type=float, default=1)
parser.add_argument('--disc_iters', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--data_root', type=str, default='data/celeba_wild')
parser.add_argument('--class_name', type=str, default='celeba_wild')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--n_embedding', type=int, default=64)
parser.add_argument('--checkpoint', type=int, default=0)
args = parser.parse_args()

args.log = 'gan_{0}_k{1}'.format(args.class_name, args.n_keypoints)
args.log = os.path.join('log', args.log)

os.makedirs(args.log, exist_ok=True)
if args.checkpoint == 0:
    with open(os.path.join(args.log, 'parameters.json'), 'wt') as f:
        json.dump(args.__dict__, f, indent=2)
else:
    with open(os.path.join(args.log, 'parameters.json'), 'rt') as f:
        t_args = argparse.Namespace()
        old_para = json.load(f)
        old_para.update(args.__dict__)
        t_args.__dict__.update(old_para)
        args = parser.parse_args(namespace=t_args)

data_loader = get_dataloader(data_root=args.data_root, class_name=args.class_name,
                             image_size=args.image_size, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
model = importlib.import_module('models.' + args.model)
generator = model.Generator({'z_dim': args.z_dim, 'n_keypoints': args.n_keypoints, 'n_per_kp': args.n_per_kp,
                             'n_embedding': args.n_embedding, 'image_size': args.image_size,
                             'single_final': args.single_final, 'use_linear': args.use_linear,
                             'smaller_init_mask': args.smaller_init_mask, 'feature_map_sizes': args.feature_map_sizes,
                             'feature_map_channels': args.feature_map_channels,
                             'class_name': args.class_name}).to(device)
dis_model = importlib.import_module('models.' + args.dis_model)
discriminator = dis_model.Discriminator({}).to(device)
optim_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(0.5, 0.9))
optim_gen = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(0.5, 0.9))

generator = torch.nn.DataParallel(generator)
discriminator = torch.nn.DataParallel(discriminator)


kp_color, pt_color = get_pt_color(args.n_keypoints, args.n_per_kp)


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch.autograd.grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_one_epoch():
    discriminator.train()
    generator.train()
    total_disc_loss = 0
    total_gen_loss = 0

    for batch_index, real_batch in enumerate(data_loader):
        optim_disc.zero_grad()
        optim_gen.zero_grad()

        # update discriminator
        real_batch = {'img': real_batch['img'].to(device)}
        real_batch['img'].requires_grad_()
        input_batch = {'input_noise{}'.format(noise_i): torch.randn(args.batch_size, *noise_shape).to(device)
                       for noise_i, noise_shape in enumerate(generator.module.noise_shapes)}
        input_batch['bg_trans'] = torch.rand(args.batch_size, 1, 2).to(device) * 2 - 1
        fake_batch = generator(input_batch)
        d_real_out = discriminator(real_batch)
        d_fake_out = discriminator(fake_batch)
        disc_loss = F.softplus(d_fake_out).mean() + F.softplus(-d_real_out).mean()
        if batch_index % 16 == 0:
            disc_loss = disc_loss + gradient_penalty(real_batch['img'], d_real_out)
        # print(disc_loss)
        disc_loss.backward()
        total_disc_loss += disc_loss.item()
        optim_disc.step()

        # update generator
        if batch_index % args.disc_iters == 0:
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            input_batch = {'input_noise{}'.format(noise_i): torch.randn(args.batch_size, *noise_shape).to(device)
                           for noise_i, noise_shape in enumerate(generator.module.noise_shapes)}
            input_batch['bg_trans'] = torch.rand(args.batch_size, 1, 2).to(device) * 2 - 1
            fake_batch = generator(input_batch, requires_penalty=True)
            d_fake_out = discriminator(fake_batch)
            gen_loss = F.softplus(-d_fake_out).mean()
            if batch_index % 2 == 0:
                gen_loss = gen_loss + fake_batch['center_penalty'].mean() * args.con_penalty_coef + fake_batch['area_penalty'].mean() * args.area_penalty_coef
            gen_loss.backward()
            total_gen_loss += gen_loss.item()
            optim_gen.step()
            # break

        if batch_index > 1000:
            break

    return total_disc_loss / args.disc_iters / len(data_loader) / 2, total_gen_loss / len(data_loader)


def evaluate(test_input_batch):
    eval_dir = os.path.join(args.log, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        sample_batch = generator(test_input_batch)
        samples = sample_batch['img'][:64].cpu().numpy()
        keypoints = sample_batch['keypoints'][:64].cpu().numpy() * (args.image_size / 2 - 0.5) + (args.image_size / 2 - 0.5)
        kp_mask = sample_batch['kp_mask'][:64].cpu()
        bg_mask = sample_batch['bg_mask'][:64].cpu()
        color_mask = torch.cat([bg_mask, kp_mask], dim=1).max(dim=1)[1].cpu().numpy().astype(int)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.1, hspace=0.1)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)

    plt.savefig(os.path.join(eval_dir, '{}.png'.format(epoch)), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.1, hspace=0.1)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)
        ax.scatter(keypoints[i, :, 1], keypoints[i, :, 0], c=kp_color, s=20, marker='+')

    plt.savefig(os.path.join(eval_dir, '{}_keypoints.png'.format(epoch)), bbox_inches='tight')
    plt.close(fig)

    if 'points' in sample_batch.keys():
        points = sample_batch['points'][:64].cpu().numpy() * (args.image_size / 2 - 0.5) + (args.image_size / 2 - 0.5)
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.1, hspace=0.1)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)
            ax.scatter(points[i, :, 1], points[i, :, 0], c=pt_color, s=20, marker='+')

        plt.savefig(os.path.join(eval_dir, '{}_points.png'.format(epoch)), bbox_inches='tight')
        plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.1, hspace=0.1)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(label2rgb(color_mask[i], sample.transpose((1, 2, 0)) * 0.5 + 0.5, segment_color=kp_color, alpha=0.5))

    plt.savefig(os.path.join(eval_dir, '{}_segmaps.png'.format(epoch)), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(args.n_keypoints+1, args.n_keypoints+1))
    gs = gridspec.GridSpec(args.n_keypoints+1, args.n_keypoints+1)
    gs.update(wspace=0.1, hspace=0.1)

    for i in range(min(args.batch_size, args.n_keypoints+1)):
        for j in range(args.n_keypoints):
            ax = plt.subplot(gs[i*(args.n_keypoints+1)+j])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(kp_mask[i, j], vmin=0, vmax=1)
            plt.scatter(keypoints[i, j, 1], keypoints[i, j, 0], s=20, marker='+')
        ax = plt.subplot(gs[i * (args.n_keypoints + 1) + args.n_keypoints])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(bg_mask[i, 0], vmin=0, vmax=1)
        plt.scatter(keypoints[i, :, 1], keypoints[i, :, 0], s=20, marker='+')

    plt.savefig(os.path.join(eval_dir, '{}_heatmaps.png'.format(epoch)), bbox_inches='tight')
    plt.close(fig)

    if 'init_mask' in sample_batch.keys():
        kp_init_mask = sample_batch['init_mask'][:64].cpu()
        bg_init_mask = 1 - kp_init_mask[:64].max(dim=1, keepdim=True)[0]
        color_init_mask = torch.cat([bg_init_mask * 0.1, kp_init_mask], dim=1).max(dim=1)[1].cpu().numpy().astype(int)

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.1, hspace=0.1)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(label2rgb(color_init_mask[i], sample.transpose((1, 2, 0)) * 0.5 + 0.5, segment_color=kp_color, alpha=0.5))

        plt.savefig(os.path.join(eval_dir, '{}_init_mask.png'.format(epoch)), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    writer = SummaryWriter(os.path.join(args.log, 'runs'))
    test_input_batch = {'input_noise{}'.format(noise_i): torch.randn(args.batch_size, *noise_shape).to(device)
                        for noise_i, noise_shape in enumerate(generator.module.noise_shapes)}
    test_input_batch['bg_trans'] = torch.rand(args.batch_size, 1, 2).to(device) * 2 - 1
    checkpoint_dir = os.path.join(args.log, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.checkpoint != 0:
        checkpoint = torch.load(os.path.join(args.log, 'checkpoints', 'epoch_{}.model'.format(args.checkpoint)),
                                map_location=lambda storage, location: storage)
        generator.module.load_state_dict(checkpoint['generator'])
        discriminator.module.load_state_dict(checkpoint['discriminator'])
        optim_gen.load_state_dict(checkpoint['optim_gen'])
        optim_disc.load_state_dict(checkpoint['optim_disc'])
        args.checkpoint += 1

    for epoch in range(args.checkpoint, 30):
        disc_loss, gen_loss = train_one_epoch()
        writer.add_scalars('loss', {'disc_loss': disc_loss,
                                    'gen_loss': gen_loss}, epoch + 1)

        evaluate(test_input_batch)

        if (epoch + 1) % 1 == 0:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'optim_gen': optim_gen.state_dict(),
                    'optim_disc': optim_disc.state_dict(),
                },
                os.path.join(checkpoint_dir, 'epoch_{}.model'.format(epoch))
            )
