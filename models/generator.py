import torch
from torch import nn
import torch.nn.functional as F
from models.heatmaps import gen_heatmaps_extend, gen_heatmaps_linear_extend, gen_grid2d_extend, gen_grid2d
from models.sync_batchnorm import SynchronizedBatchNorm2d
import math
import numpy as np


class SPADE(nn.Module):
    def __init__(self, input_channel, n_embeddings, n_keypoints):
        super().__init__()
        self.norm = SynchronizedBatchNorm2d(input_channel, affine=False)
        self.conv = nn.Conv2d(n_embeddings, 128, kernel_size=3, padding=1)
        self.conv_gamma = nn.Conv2d(128, input_channel, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, input_channel, kernel_size=3, padding=1)

    def forward(self, x, heatmaps):
        normalized_x = self.norm(x)
        heatmaps_features = F.leaky_relu(self.conv(heatmaps), 0.2)
        heatmaps_gamma = self.conv_gamma(heatmaps_features)
        heatmaps_beta = self.conv_beta(heatmaps_features)
        return (1+heatmaps_gamma) * normalized_x + heatmaps_beta


class SPADEResBlk(nn.Module):
    def __init__(self, in_channel, out_channel, n_embeddings, n_keypoints):
        super().__init__()
        mid_channel = min(in_channel, out_channel)
        self.learn_shortcut = in_channel != out_channel
        self.spade1 = SPADE(in_channel, n_embeddings, n_keypoints)
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)
        self.spade2 = SPADE(mid_channel, n_embeddings, n_keypoints)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)

        if self.learn_shortcut:
            self.spade_shortcut = SPADE(in_channel, n_embeddings, n_keypoints)
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x, heatmaps):
        shortcut = x
        x = self.conv1(F.leaky_relu(self.spade1(x, heatmaps), 0.2))
        x = self.conv2(F.leaky_relu(self.spade2(x, heatmaps), 0.2))

        if self.learn_shortcut:
            shortcut = self.conv_shortcut(self.spade_shortcut(shortcut, heatmaps))

        return x + shortcut


class AdaIN(nn.Module):
    def __init__(self, input_channel, n_embeddings):
        super().__init__()
        self.norm = SynchronizedBatchNorm2d(input_channel, affine=False)
        self.conv = nn.Linear(n_embeddings, 128)
        self.conv_gamma = nn.Linear(128, input_channel)
        self.conv_beta = nn.Linear(128, input_channel)

    def forward(self, x, style):
        normalized_x = self.norm(x)
        style = F.leaky_relu(self.conv(style), 0.2)
        gamma = self.conv_gamma(style).unsqueeze(-1).unsqueeze(-1)
        beta = self.conv_beta(style).unsqueeze(-1).unsqueeze(-1)
        return (1+gamma) * normalized_x + beta


class AdaINResBlk(nn.Module):
    def __init__(self, in_channel, out_channel, n_embeddings):
        super().__init__()
        self.spade1 = AdaIN(in_channel, n_embeddings)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x, style):
        x = self.conv1(F.leaky_relu(self.spade1(x, style), 0.2))

        return x


class Conv2dLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2dLeakyReLU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.model(x)


class LinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLeakyReLU, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.model(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, n_keypoints, out_channels):
        super().__init__()
        self.pe = nn.Conv2d(2*n_keypoints, out_channels // 2, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pe(x) * math.pi
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=1)
        return x


class Generator(nn.Module):
    def __init__(self, hyper_paras):
        super(Generator, self).__init__()
        self.z_dim = hyper_paras['z_dim']
        self.n_keypoints = hyper_paras['n_keypoints']
        self.n_points_per_kp = hyper_paras['n_per_kp']
        self.n_points = self.n_points_per_kp * self.n_keypoints
        self.n_embedding = hyper_paras['n_embedding']
        self.use_linear = hyper_paras['use_linear']
        self.smaller_init_mask = hyper_paras['smaller_init_mask']
        self.image_size = hyper_paras['image_size']
        self.feature_map_sizes = hyper_paras['feature_map_sizes']
        self.feature_map_channels = hyper_paras['feature_map_channels']
        self.single_final = hyper_paras['single_final']
        # self.point_divisor = 5  # VERY IMPORTANT MAGIC NUMBER, HARD CODED (also works)
        # self.point_divisor = 10  # VERY IMPORTANT MAGIC NUMBER, HARD CODED (also works)
        self.point_divisor = 20  # VERY IMPORTANT MAGIC NUMBER, HARD CODED (also works)

        self.feature_map_sizes = self.feature_map_sizes.split(',')
        self.feature_map_sizes = [int(map_size) for map_size in self.feature_map_sizes]

        self.feature_map_channels = self.feature_map_channels.split(',')
        self.feature_map_channels = [int(map_channel) for map_channel in self.feature_map_channels]

        if self.use_linear:
            self.gen_heatmap = gen_heatmaps_linear_extend
        else:
            self.gen_heatmap = gen_heatmaps_extend

        self.extend_pixel = 10

        self.noise_shapes = [(self.z_dim,), (self.z_dim,), (self.z_dim,)]

        self.keypoints_embedding = nn.Embedding(self.n_keypoints, self.n_embedding)

        # self.gen_keypoints_embedding_noise = nn.Sequential(
        #     *([LinearLeakyReLU(self.z_dim, self.z_dim)] * 3),
        #     nn.Linear(self.z_dim, self.n_embedding),
        # )
        #
        # self.gen_keypoints_layer = nn.Sequential(
        #     *([LinearLeakyReLU(self.z_dim, self.z_dim)] * 4),
        #     nn.Linear(self.z_dim, self.n_points * 2),
        #     )
        #
        # self.gen_background_embedding = nn.Sequential(
        #     *([LinearLeakyReLU(self.z_dim, self.z_dim)] * 3),
        #     nn.Linear(self.z_dim, self.n_embedding),
        # )
        # I don't know why there seems a bug in the above code. It looks equivalent to the code below.

        self.gen_keypoints_embedding_noise = nn.Sequential(
            LinearLeakyReLU(self.z_dim, self.z_dim),
            LinearLeakyReLU(self.z_dim, self.z_dim),
            LinearLeakyReLU(self.z_dim, self.z_dim),
            nn.Linear(self.z_dim, self.n_embedding),
        )

        self.gen_keypoints_layer = nn.Sequential(
            LinearLeakyReLU(self.z_dim, self.z_dim),
            LinearLeakyReLU(self.z_dim, self.z_dim),
            LinearLeakyReLU(self.z_dim, self.z_dim),
            LinearLeakyReLU(self.z_dim, self.z_dim),
            nn.Linear(self.z_dim, self.n_points * 2),
        )

        self.gen_background_embedding = nn.Sequential(
            LinearLeakyReLU(self.z_dim, self.z_dim),
            LinearLeakyReLU(self.z_dim, self.z_dim),
            LinearLeakyReLU(self.z_dim, self.z_dim),
            nn.Linear(self.z_dim, self.n_embedding),
        )

        self.x_start = PositionalEmbedding(n_keypoints=self.n_keypoints, out_channels=self.feature_map_channels[0])
        self.mask_start = PositionalEmbedding(n_keypoints=self.n_points, out_channels=self.feature_map_channels[0])
        self.bg_start = PositionalEmbedding(n_keypoints=1, out_channels=self.feature_map_channels[0])

        self.rep_pad = nn.ReplicationPad2d(self.extend_pixel)

        self.mask_spade_blocks = nn.ModuleList([
            SPADEResBlk(self.feature_map_channels[0], self.n_embedding, self.n_embedding, self.n_keypoints - 1),
            *([SPADEResBlk(self.n_embedding, self.n_embedding, self.n_embedding, self.n_keypoints - 1)] *
              (len(self.feature_map_sizes) - 2)),
            SPADEResBlk(self.n_embedding, self.n_keypoints+1, self.n_embedding, self.n_keypoints - 1)
        ])

        self.spade_blocks = nn.ModuleList([])
        for i in range(len(self.feature_map_channels) - 1):
            self.spade_blocks.append(SPADEResBlk(self.feature_map_channels[i], self.feature_map_channels[i + 1],
                                                 self.n_embedding, self.n_keypoints - 1))
        self.spade_blocks.append(SPADEResBlk(self.feature_map_channels[-1], self.n_embedding,
                                             self.n_embedding, self.n_keypoints - 1))

        self.adain_blocks = nn.ModuleList([])
        for i in range(len(self.feature_map_channels) - 1):
            self.adain_blocks.append(AdaINResBlk(self.feature_map_channels[i], self.feature_map_channels[i + 1],
                                                 self.n_embedding))
        self.adain_blocks.append(AdaINResBlk(self.feature_map_channels[-1], self.n_embedding,
                                             self.n_embedding))

        if self.single_final:
            self.conv = nn.Sequential(
                nn.Conv2d(self.n_embedding, 3, kernel_size=1, padding=0),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.n_embedding, self.n_embedding, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.n_embedding, 3, kernel_size=3, padding=1),
            )

        grid = gen_grid2d_extend(self.feature_map_sizes[0], extend=self.extend_pixel).reshape(1, -1, 2)
        self.init_extend_coord = nn.Parameter(grid, requires_grad=False)

        self.coord = {}

        for image_size in np.unique(self.feature_map_sizes):
            self.coord[str(image_size)] = gen_grid2d(image_size).reshape(1, -1, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input_dict, requires_penalty=False):
        out_batch = self.gen_keypoints(input_dict)
        out_batch = self.gen_mask(out_batch)
        out_batch = self.gen_foreground(out_batch)
        out_batch = self.gen_background(out_batch)
        out_batch = self.gen_img(out_batch)

        center_penalty = torch.tensor(0.0, device=out_batch['img'].device)
        area_penalty = torch.tensor(0.0, device=out_batch['img'].device)

        if requires_penalty:
            kp_mask = out_batch['kp_mask']
            coord = self.coord[str(kp_mask.shape[-1])].to(kp_mask)
            diff = out_batch['keypoints'].unsqueeze(-2) - coord.unsqueeze(-3)  # (batch_size, n_points, heatmap_size**2, 2)
            dist = diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2
            normalized_weight = kp_mask / (kp_mask.sum(dim=(-1, -2), keepdim=True) + 1e-6)
            weighted_dist = dist * normalized_weight.reshape(dist.shape[0], self.n_keypoints, -1)
            weighted_dist = weighted_dist.sum(dim=-1)
            center_penalty = center_penalty + weighted_dist.mean()

            if self.smaller_init_mask:
                mask_area = kp_mask.sum(dim=(-1, -2))
                init_mask_area = out_batch['init_mask'].sum(dim=(-1, -2))
                area_penalty = area_penalty + torch.clamp(init_mask_area-mask_area, min=0).sum(dim=1).mean()
            else:
                mask_area = kp_mask.sum(dim=(-1, -2))
                tau = out_batch['points'].reshape(-1, self.n_keypoints, self.n_points_per_kp, 2) - out_batch['keypoints'].reshape(-1, self.n_keypoints, 1, 2)
                tau = tau.abs().mean(dim=(2, 3)) * (self.image_size / 2)
                init_mask_area = math.pi * tau**2
                area_penalty = area_penalty + torch.clamp(init_mask_area-mask_area, min=0).sum(dim=1).mean()

        out_batch['center_penalty'] = center_penalty
        out_batch['area_penalty'] = area_penalty

        return out_batch

    def crop(self, x):
        return x[:, :, self.extend_pixel:-self.extend_pixel, self.extend_pixel:-self.extend_pixel]

    def gen_keypoints(self, input_dict):
        # generate location
        z = input_dict['input_noise0']
        points = torch.tanh(self.gen_keypoints_layer(z).reshape(-1, self.n_points, 2) / self.point_divisor)
        keypoints = points.reshape(points.shape[0], self.n_keypoints, -1, 2).mean(dim=2)

        # generate feature
        kp_embed_noise = input_dict['input_noise2']
        kp_fixed_emb = self.keypoints_embedding(
            torch.arange(self.n_keypoints, device=kp_embed_noise.device).unsqueeze(0).repeat(kp_embed_noise.shape[0], 1)
        )
        kp_emb = self.gen_keypoints_embedding_noise(kp_embed_noise)
        kp_emb = kp_fixed_emb * kp_emb.unsqueeze(1)

        input_dict['points'] = points
        input_dict['keypoints'] = keypoints
        input_dict['kp_emb'] = kp_emb

        return input_dict

    def gen_mask(self, input_dict):
        keypoints = input_dict['keypoints']
        points = input_dict['points']
        kp_emb = input_dict['kp_emb']

        diff = points.unsqueeze(-2) - self.init_extend_coord.unsqueeze(-3)  # (batch_size, n_points, self.image_sizes[0]**2, 2)
        diff = diff.transpose(2, 3).contiguous()  # (batch_size, n_points, 2, self.image_sizes[0]**2)
        diff = diff.reshape(-1, 2 * self.n_points, (self.feature_map_sizes[0] + 2 * self.extend_pixel), (self.feature_map_sizes[0] + 2 * self.extend_pixel))
        mask = self.mask_start(diff)

        if self.use_linear:
            tau = points.reshape(-1, self.n_keypoints, self.n_points_per_kp, 2) - keypoints.reshape(-1, self.n_keypoints, 1, 2)
            tau = tau.abs().mean(dim=(2, 3)) + 1e-6
            tau = tau.reshape(-1, self.n_keypoints, 1)
        else:
            tau = points.reshape(-1, self.n_keypoints, self.n_points_per_kp, 2) - keypoints.reshape(-1, self.n_keypoints, 1, 2)
            tau = tau.norm(dim=-1).var(dim=-1, keepdim=True) + 1e-6

        for i in range(len(self.feature_map_sizes)):
            init_mask = self.gen_heatmap(keypoints, self.feature_map_sizes[i], tau=tau, extend=self.extend_pixel)
            heatmaps = init_mask.unsqueeze(2) * kp_emb.unsqueeze(-1).unsqueeze(-1)
            heatmaps = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], kp_emb.shape[2], mask.shape[-1], mask.shape[-1]).sum(dim=1)
            mask = self.mask_spade_blocks[i](mask, heatmaps)
            if i == len(self.feature_map_sizes) - 1:
                break
            elif self.feature_map_sizes[i] != self.feature_map_sizes[i + 1]:
                mask = self.crop(F.interpolate(mask, size=mask.shape[-1]*2, mode='bilinear', align_corners=False))

        mask = F.softmax(self.crop(mask), dim=1)

        input_dict['mask'] = mask
        input_dict['init_mask'] = self.crop(init_mask)

        return input_dict

    def gen_foreground(self, input_dict):
        keypoints = input_dict['keypoints']
        kp_emb = input_dict['kp_emb']
        mask = input_dict['mask']

        diff = keypoints.unsqueeze(-2) - self.init_extend_coord.unsqueeze(-3)  # (batch_size, n_points, self.image_sizes[0]**2, 2)
        diff = diff.transpose(2, 3).contiguous()  # (batch_size, n_points, 2, self.image_sizes[0]**2)
        fg = diff.reshape(-1, 2 * self.n_keypoints, (self.feature_map_sizes[0] + 2 * self.extend_pixel), (self.feature_map_sizes[0] + 2 * self.extend_pixel))
        fg = self.x_start(fg)

        for i in range(len(self.feature_map_sizes)):
            current_kp_mask = self.rep_pad(F.interpolate(mask[:, :-1, :, :], size=self.feature_map_sizes[i], mode='bilinear', align_corners=False))
            heatmaps = current_kp_mask.unsqueeze(2) * kp_emb.unsqueeze(-1).unsqueeze(-1)
            heatmaps = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], kp_emb.shape[2], fg.shape[-1], fg.shape[-1]).sum(dim=1)
            fg = self.spade_blocks[i](fg, heatmaps)

            if i == len(self.feature_map_sizes) - 1:
                break

            elif self.feature_map_sizes[i] != self.feature_map_sizes[i + 1]:
                fg = self.crop(F.interpolate(fg, size=fg.shape[-1]*2, mode='bilinear', align_corners=False))

        input_dict['fg'] = self.crop(fg)

        return input_dict

    def gen_background(self, input_dict):
        bg_center = input_dict['bg_trans']
        bg_center = torch.cat([torch.zeros_like(bg_center[:, :, 0:1]), bg_center[:, :, 1:2]], dim=2)
        bg_emb = self.gen_background_embedding(input_dict['input_noise1'])

        diff = bg_center.unsqueeze(-2) - self.init_extend_coord.unsqueeze(-3)  # (batch_size, 1, self.image_sizes[0]**2, 2)
        diff = diff.transpose(2, 3).contiguous()  # (batch_size, 1, 2, self.image_sizes[0]**2)
        bg = diff.reshape(-1, 2, (self.feature_map_sizes[0] + 2 * self.extend_pixel), (self.feature_map_sizes[0] + 2 * self.extend_pixel))
        bg = self.bg_start(bg)

        for i, adain_block in enumerate(self.adain_blocks):
            bg = adain_block(bg, bg_emb)
            if self.feature_map_sizes[i] != self.feature_map_sizes[min(i + 1, len(self.feature_map_sizes) - 1)]:
                bg = self.crop(F.interpolate(bg, size=bg.shape[-1] * 2, mode='bilinear', align_corners=False))

        input_dict['bg'] = self.crop(bg)

        return input_dict

    def gen_img(self, input_dict):
        bg = input_dict['bg']
        fg = input_dict['fg']
        mask = input_dict['mask']

        kp_mask, bg_mask = mask[:, :-1, :, :], mask[:, -1:, :, :]
        img = (1 - bg_mask) * fg + bg_mask * bg
        img = torch.tanh(self.conv(F.leaky_relu(img, 0.2)))

        input_dict['img'] = img
        input_dict['kp_mask'] = kp_mask
        input_dict['bg_mask'] = bg_mask

        return input_dict


if __name__ == '__main__':
    model = Generator({'z_dim': 256, 'n_keypoints': 10, 'n_embedding': 128, 'tau': 0.01})
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
