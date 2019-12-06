"""
Contains various differentiable image transforms.
Loosely based on Lucid's transforms.py https://github.com/tensorflow/lucid/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import kornia


class RandomScale(nn.Module):
  def __init__(self, scales):
    super(RandomScale, self).__init__()

    self.scales = scales

  def forward(self, x: torch.Tensor):
    scale = self.scales[random.randint(0, len(self.scales)-1)]
    return F.interpolate(x, scale_factor=scale, mode='bilinear')


class RandomCrop(nn.Module):
  def __init__(self, size: int):
    super(RandomCrop, self).__init__()
    self.size = size

  def forward(self, x: torch.Tensor):
    batch_size, _, h, w = x.shape
    h_move = random.randint(0, self.size)
    w_move = random.randint(0, self.size)
    return x[:, :, h_move:h-self.size+h_move, w_move:w-self.size+w_move]


class RandomRotate(nn.Module):
  def __init__(self, angle=10, same_throughout_batch=False):
    super(RandomRotate, self).__init__()
    self.angle=angle
    self.same_throughout_batch = same_throughout_batch

  def forward(self, img: torch.tensor):
    b, _, h, w = img.shape
    # create transformation (rotation)
    if not self.same_throughout_batch:
      angle = torch.randn(b, device=img.device) * self.angle
    else:
      angle = torch.randn(1, device=img.device) * self.angle
      angle = angle.repeat(b)
    center = torch.ones(b, 2, device=img.device)
    center[..., 0] = img.shape[3] / 2  # x
    center[..., 1] = img.shape[2] / 2  # y
    # define the scale factor
    scale = torch.ones(b, device=img.device)
    M = kornia.get_rotation_matrix2d(center, angle, scale)
    img_warped = kornia.warp_affine(img, M, dsize=(h, w))
    return img_warped


class Normalization(nn.Module):
  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    # .view the mean and std to make them [C x 1 x 1] so that they can
    # directly work with image Tensor of shape [B x C x H x W].
    # B is batch size. C is number of channels. H is height and W is width.
    self.mean = torch.tensor(mean).view(-1, 1, 1)
    self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
    # normalize img
    return (img - self.mean) / self.std
