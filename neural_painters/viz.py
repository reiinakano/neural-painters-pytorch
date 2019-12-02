import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List,Union

import moviepy.editor as mpy


def plot_images(images, figsize=(16, 16)):
  fig=plt.figure(figsize=figsize)
  columns = len(images)

  for i, img in enumerate(images):
    img = img[:, :, :3]
    fig.add_subplot(1, columns, i+1)
    plt.grid(False)
    plt.imshow(img)
  plt.show()


def animate_frames(frames, video_path):
  def frame(t):
    t = int(t * 10.)
    if t >= len(frames):
      t = len(frames) - 1
    return frames[t]

  clip = mpy.VideoClip(frame, duration=len(frames) // 10)
  clip.write_videofile(video_path, fps=10.)


def validate_neural_painter(strokes, actions, neural_painter, checkpoints_to_test):
  for ckpt in checkpoints_to_test:
    neural_painter.load_from_train_checkpoint(ckpt)
    with torch.no_grad():
      pred_strokes = neural_painter(actions)

    plot_images(np.transpose(strokes.numpy(), [0, 2, 3, 1]))
    plot_images(np.transpose(pred_strokes.numpy(), [0, 2, 3, 1]))


def neural_painter_stroke_animation(neural_painter_fn,
                                    action_size,
                                    checkpoints_to_test,
                                    video_path,
                                    num_acs=8,
                                    duration=10.0,
                                    fps=30.0,
                                    real_env=None):
  if real_env:
    real_env.reset()
  acs = np.random.uniform(size=[num_acs, action_size])

  neural_painters = []
  for ckpt in checkpoints_to_test:
    x = neural_painter_fn()
    x.load_from_train_checkpoint(ckpt)
    neural_painters.append(x)

  def frame(t):
    t_ = t / duration
    t = np.abs((1.0 - np.cos(num_acs * np.pi * np.mod(t_, 1. / num_acs))) / 2.0)

    new_ac = (1 - t) * acs[int(np.floor(t_ * num_acs))] + t * acs[int((np.floor(t_ * num_acs) + 1) % num_acs)]
    if real_env:
      real_env.draw(new_ac)
      im = real_env.image
      im = im[:, :, :3]
    stack_these = []
    for neural_painter in neural_painters:
      with torch.no_grad():
        decoded = neural_painter(torch.FloatTensor([new_ac]))
      decoded = np.transpose(decoded.numpy(), [0, 2, 3, 1])[0]
      decoded = (decoded * 255).astype(np.uint8)
      if real_env:
        decoded = np.concatenate([im, decoded], 1)
      stack_these.append(decoded)
    return np.concatenate(stack_these, axis=0)

  clip = mpy.VideoClip(frame, duration=duration)
  clip.write_videofile(video_path, fps=fps)
  print("written to {}".format(video_path))


def animate_strokes_on_canvas(intermediate_canvases: List[torch.Tensor],
                              target_image: Union[torch.tensor, None],
                              video_path: str, skip_every_n: int = 1,
                              batch_idx: int = 0):
  _, _, h, w = intermediate_canvases[0].shape

  # We take only one sample of the batch for each step and concatenate everything into a numpy array
  intermediate_canvases = [(x.detach().cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)[batch_idx]
                           for x in intermediate_canvases]
  intermediate_canvases = np.stack(intermediate_canvases)
  intermediate_canvases = intermediate_canvases[::skip_every_n]

  to_plot = intermediate_canvases

  if target_image is not None:
    target_images = (target_image.detach().cpu().numpy() * 255).astype(np.uint8).reshape(1, 3, h, w).transpose(0, 2, 3, 1)
    target_images = np.tile(target_images, [len(intermediate_canvases), 1, 1, 1])
    to_plot = np.concatenate([target_images, to_plot], axis=(2 if h >= w else 1))

  to_plot = np.concatenate([to_plot, np.tile(to_plot[-1:, :, :, :], [50, 1, 1, 1])], axis=0)

  animate_frames(to_plot, video_path)
