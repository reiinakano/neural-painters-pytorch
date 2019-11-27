import torch
import torch.nn as nn

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

  def forward(self, actions: torch.Tensor):
    """
    actions: tensor of shape (num_strokes, batch_size, action_size)
    """
    num_strokes, batch_size, action_size = actions.shape

    intermediate_canvases = []
    next_canvas = torch.ones(batch_size, 3, self.final_canvas_height, self.final_canvas_width).to(actions.device)
    intermediate_canvases.append(next_canvas)
    for i in range(num_strokes):
      stroke = self.neural_painter(actions[i])
      next_canvas = paint_over_canvas(next_canvas, stroke, actions[i, :, 6:9])
      intermediate_canvases.append(next_canvas)

    return next_canvas, intermediate_canvases


class RandomRotate(nn.Module):
  def __init__(self, angle=10):
    super(RandomRotate, self).__init__()
    self.angle=angle

  def forward(self, img: torch.tensor):
    b, _, h, w = img.shape
    # create transformation (rotation)
    angle = torch.randn(b, device=img.device) * self.angle
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
