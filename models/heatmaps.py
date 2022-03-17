import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


def gen_grid2d(grid_size, device='cpu', left_end=-1, right_end=1):
    x = torch.linspace(left_end, right_end, grid_size).to(device)
    x, y = torch.meshgrid([x, x])
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid


def gen_grid2d_extend(grid_size, device='cpu', extend=10, left_end=-1, right_end=1):
    x = torch.linspace(left_end, right_end, grid_size)
    right_end = 1 + (x[1] - x[0]) * extend
    heatmap_size = grid_size + 2 * extend
    x = torch.linspace(-right_end, right_end, heatmap_size).to(device)
    x, y = torch.meshgrid([x, x])
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(heatmap_size, heatmap_size, 2)
    return grid


def gen_heatmaps(input, heatmap_size=16, tau=0.01):
    """
    :param input: (batch_size, n_points, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    """
    batch_size, n_points, _ = input.shape
    grid = gen_grid2d(heatmap_size, device=input.device).reshape(1, -1, 2).expand(batch_size, -1, -1)
    input_norm = (input ** 2).sum(dim=2).unsqueeze(2)
    grid_norm = (grid ** 2).sum(dim=2).unsqueeze(1)
    dist = input_norm + grid_norm - 2 * torch.bmm(input, grid.permute(0, 2, 1))
    heatmaps = torch.exp(-dist / tau)
    return heatmaps.reshape(batch_size, n_points, heatmap_size, heatmap_size)


def gen_heatmaps_extend(input, heatmap_size=16, tau=0.01, extend=1):
    """
    :param input: (batch_size, n_points, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    """
    batch_size, n_points, _ = input.shape
    grid = gen_grid2d_extend(heatmap_size, device=input.device, extend=extend).reshape(1, -1, 2).expand(batch_size, -1, -1)
    input_norm = (input ** 2).sum(dim=2).unsqueeze(2)
    grid_norm = (grid ** 2).sum(dim=2).unsqueeze(1)
    dist = input_norm + grid_norm - 2 * torch.bmm(input, grid.permute(0, 2, 1))
    heatmaps = torch.exp(-dist / tau)
    heatmap_size = heatmap_size + 2 * extend
    return heatmaps.reshape(batch_size, n_points, heatmap_size, heatmap_size)


def gen_heatmaps_linear(input, heatmap_size=16, tau=0.01, map_range=1):
    """
    :param input: (batch_size, n_points, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    """
    batch_size, n_points, _ = input.shape
    grid = gen_grid2d(heatmap_size, device=input.device).reshape(1, -1, 2).expand(batch_size, -1, -1)
    diff = input.unsqueeze(-2) - grid.unsqueeze(-3)  # (batch_size, n_points, 4**2, 2)
    dist = diff.norm(dim=-1)  # (batch_size, n_points, 2, 4**2)
    heatmaps = torch.clamp(1 - dist / tau, min=0)
    return heatmaps.reshape(batch_size, n_points, heatmap_size, heatmap_size)


def gen_heatmaps_linear_extend(input, heatmap_size=16, tau=0.15, extend=1):
    """
    :param input: (batch_size, n_points, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    """
    batch_size, n_points, _ = input.shape
    grid = gen_grid2d_extend(heatmap_size, device=input.device, extend=extend).reshape(1, -1, 2).expand(batch_size, -1, -1)
    diff = input.unsqueeze(-2) - grid.unsqueeze(-3)  # (batch_size, n_points, 4**2, 2)
    dist = diff.norm(dim=-1)  # (batch_size, n_points, 2, 4**2)
    heatmaps = torch.clamp(1 - dist / tau, min=0)
    heatmap_size = heatmap_size + 2 * extend
    return heatmaps.reshape(batch_size, n_points, heatmap_size, heatmap_size)
