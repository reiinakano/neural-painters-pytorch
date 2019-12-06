"""
Contains canvases, containing logic for blending multiple strokes from a neural painter into one final canvas.
"""
import torch
import torch.nn as nn


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
  def __init__(self, neural_painter, action_preprocessor=torch.sigmoid):
    """
    neural_painter: Neural painter to wrap
    action_preprocessor: Set the action preprocessor for this canvas. It is called on the input tensor before passing on
    to the actual neural painter. This is where one can specify manual constraints on actions e.g. grayscale strokes,
    controlling stroke thickness, etc. By default this canvas uses torch.sigmoid to make sure input actions are in the
    range [0, 1] i.e. backprop can make the input actions go beyond this range. We suggest you call sigmoid() somewhere
    as well if you plan to use your own action preprocessor.
    """
    super(NeuralCanvas, self).__init__()

    self.neural_painter = neural_painter
    self.canvas_height = 64
    self.canvas_width = 64
    self.final_canvas_height = 64
    self.final_canvas_width = 64

    self.action_preprocessor = action_preprocessor

  def forward(self, actions: torch.Tensor):
    """
    actions: tensor of shape (num_strokes, batch_size, action_size)

    Returns tuple of:
      final_canvas: Image tensor of shape (batch_size, height, width). contains final canvas.
      intermediate_canvases: List of length num_strokes of image tensors of shape (batch_size, height, width).
                             Each tensor in the list represents one stroke. Used for visualization.
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


class NeuralCanvasStitched(nn.Module):
  """
  NeuralCanvasStitched is a collection of NeuralCanvas stitched together. Used to get higher resolution images from a
  low-res 64px neural painter.
  Maps a sequence of brushstrokes to a fully stitched canvas.
  """
  def __init__(self, neural_painter, overlap_px=10, repeat_h=8, repeat_w=8, strokes_per_block=5, action_preprocessor=torch.sigmoid):
    """
    neural_painter: neural painter to wrap
    overlap_px: number of overlapping pixels between canvases
    repeat_h: number of canvases stitched together for height of final canvas
    repeat_w: number of canvases stitched together for width of final canvas
    strokes_per_block: Number of strokes per canvas.
    action_preprocessor: Set the action preprocessor for this canvas. It is called on the input tensor before passing on
    to the actual neural painter. This is where one can specify manual constraints on actions e.g. grayscale strokes,
    controlling stroke thickness, etc. By default this canvas uses torch.sigmoid to make sure input actions are in the
    range [0, 1] i.e. backprop can make the input actions go beyond this range. We suggest you call sigmoid() somewhere
    as well if you plan to use your own action preprocessor.
    """
    super(NeuralCanvasStitched, self).__init__()

    self.neural_painter = neural_painter
    self.overlap_px = overlap_px
    self.repeat_h = repeat_h
    self.repeat_w = repeat_w
    self.strokes_per_block = strokes_per_block

    self.final_canvas_h = 64*repeat_h - overlap_px*(repeat_h - 1)
    self.final_canvas_w = 64*repeat_w - overlap_px*(repeat_w - 1)
    self.total_num_strokes = strokes_per_block * repeat_h * repeat_w

    self.action_preprocessor = action_preprocessor

    print(f'final canvas size H: {self.final_canvas_h} W: {self.final_canvas_w}\t'
          f'total number of strokes: {self.total_num_strokes}')

  def forward(self, actions: torch.Tensor):
    """
    actions: tensor of shape (num_strokes, batch_size, action_size)

    Returns tuple of:
      final_canvas: Image tensor of shape (batch_size, height, width). contains final canvas.
      intermediate_canvases: List of length total_num_strokes of image tensors of shape (batch_size, height, width).
                             Each tensor in the list represents one stroke. Used for visualization.
    """
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
