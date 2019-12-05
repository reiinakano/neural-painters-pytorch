import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from typing import Callable

import kornia


def paint_over_canvas(canvas: torch.Tensor, stroke: torch.Tensor, color: torch.Tensor):
  _, _, canvas_height, canvas_width = canvas.shape
  darkness_mask = torch.mean(stroke, dim=1, keepdim=True)  # Take mean over color channels
  darkness_mask = 1. - darkness_mask
  normalizer, _ = torch.max(darkness_mask, dim=2, keepdim=True)
  normalizer, _ = torch.max(normalizer, dim=3, keepdim=True)
  darkness_mask = darkness_mask / normalizer

  color_action = color.view(-1, 3, 1, 1)
  color_action = color_action.repeat(1, 1, canvas_height, canvas_width)

  blended = darkness_mask * color_action + (1-darkness_mask) * canvas
  return blended


class NeuralCanvas(nn.Module):
  """
  NeuralCanvas is a simple wrapper around a NeuralPainter. Maps a sequence of brushstrokes to a full canvas.
  Automatically performs blending.
  """
  def __init__(self, neural_painter):
    super(NeuralCanvas, self).__init__()

    self.neural_painter = neural_painter
    self.canvas_height = 64
    self.canvas_width = 64
    self.final_canvas_height = 64
    self.final_canvas_width = 64

    self.action_preprocessor = torch.sigmoid

  def forward(self, actions: torch.Tensor):
    """
    actions: tensor of shape (num_strokes, batch_size, action_size)
    """
    actions = self.action_preprocessor(actions)
    num_strokes, batch_size, action_size = actions.shape

    intermediate_canvases = []
    next_canvas = torch.ones(batch_size, 3, self.final_canvas_height, self.final_canvas_width).to(actions.device)
    intermediate_canvases.append(next_canvas.detach().cpu())
    for i in range(num_strokes):
      stroke = self.neural_painter(actions[i])
      next_canvas = paint_over_canvas(next_canvas, stroke, actions[i, :, 6:9])
      intermediate_canvases.append(next_canvas.detach().cpu())

    return next_canvas, intermediate_canvases

  def set_action_preprocessor(self, preprocessor: Callable[[torch.Tensor], torch.Tensor]):
    """
    Set the action preprocessor for this canvas. It is called on the input tensor before passing on to the
    actual neural painter. This is where one can specify manual constraints on actions e.g. grayscale strokes,
    controlling stroke thickness, etc.

    By default this canvas uses torch.sigmoid to make sure input actions are in the range [0, 1] i.e. backprop
    can make the input actions go beyond this range. We suggest you call sigmoid() somewhere as well if you plan to use
    your own action preprocessor
    """
    self.action_preprocessor = preprocessor


class NeuralCanvasStitched(nn.Module):
  """
  NeuralCanvasStitched is a collection of NeuralCanvas stitched together. Used to get higher resolution images from a
  low-res neural painter.
  Maps a sequence of brushstrokes to a fully stitched canvas.
  """
  def __init__(self, neural_painter, overlap_px=10, repeat_h=8, repeat_w=8, strokes_per_block=5):
    super(NeuralCanvasStitched, self).__init__()

    self.neural_painter = neural_painter
    self.overlap_px = overlap_px
    self.repeat_h = repeat_h
    self.repeat_w = repeat_w
    self.strokes_per_block = strokes_per_block

    self.final_canvas_h = 64*repeat_h - overlap_px*(repeat_h - 1)
    self.final_canvas_w = 64*repeat_w - overlap_px*(repeat_w - 1)
    self.total_num_strokes = strokes_per_block * repeat_h * repeat_w

    self.action_preprocessor = torch.sigmoid

    print(f'final canvas size H: {self.final_canvas_h} W: {self.final_canvas_w}\t'
          f'total number of strokes: {self.total_num_strokes}')

  def forward(self, actions: torch.Tensor):
    actions = self.action_preprocessor(actions)
    num_strokes, batch_size, action_size = actions.shape
    assert num_strokes == self.total_num_strokes

    intermediate_canvases = []
    next_canvas = torch.ones(batch_size, 3, self.final_canvas_h, self.final_canvas_w).to(actions.device)
    intermediate_canvases.append(next_canvas.detach().cpu())

    block_ctr = 0
    for a in range(self.repeat_h):
      for b in range(self.repeat_w):
        for local_stroke_idx in range(self.strokes_per_block):
          current_action = actions[block_ctr*self.strokes_per_block + local_stroke_idx]
          decoded_stroke = self.neural_painter(current_action)
          padding = nn.ConstantPad2d(
            [(64-self.overlap_px)*b,
             (64-self.overlap_px)*(self.repeat_w-1-b),
             (64-self.overlap_px)*a,
             (64-self.overlap_px)*(self.repeat_h-1-a)], 1)
          padded_stroke = padding(decoded_stroke)
          next_canvas = paint_over_canvas(next_canvas, padded_stroke, current_action[:, 6:9])
          intermediate_canvases.append(next_canvas.detach().cpu())  # Is this efficient? Maybe we should only keep it in memory in certain cases?

        block_ctr += 1
    return next_canvas, intermediate_canvases

  def set_action_preprocessor(self, preprocessor: Callable[[torch.Tensor], torch.Tensor]):
    """
    Set the action preprocessor for this canvas. It is called on the input tensor before passing on to the
    actual neural painter. This is where one can specify manual constraints on actions e.g. grayscale strokes,
    controlling stroke thickness, etc.

    By default this canvas uses torch.sigmoid to make sure input actions are in the range [0, 1] i.e. backprop
    can make the input actions go beyond this range. We suggest you call sigmoid() somewhere as well if you plan to use
    your own action preprocessor
    """
    self.action_preprocessor = preprocessor


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


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
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
