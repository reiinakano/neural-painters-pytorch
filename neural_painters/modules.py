import torch
import torch.nn as nn
import torch.nn.functional as F


def paint_over_canvas(canvas: torch.Tensor, stroke: torch.Tensor, color: torch.Tensor,
                      canvas_height: int, canvas_width: int,
                      ):
  darkness_mask = torch.mean(stroke, dim=1, keepdim=True)  # Take mean over color channels
  darkness_mask = 1. - darkness_mask
  darkness_mask = darkness_mask / torch.max(darkness_mask, dim=[2, 3], keepdim=True)

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
      next_canvas = paint_over_canvas(next_canvas, stroke, actions[i, :, 6:9],
                                      self.final_canvas_height, self.final_canvas_width)
      intermediate_canvases.append(next_canvas)

    return next_canvas, intermediate_canvases
