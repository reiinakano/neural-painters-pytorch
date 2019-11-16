import torch
import numpy as np
import matplotlib.pyplot as plt

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
                                    num_acs = 8,
                                    duration=10.0,
                                    fps=30.0):
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
    # env.draw(new_ac)
    # im = env.image
    # im = im[:, :, :3]
    stack_these = []
    for neural_painter in neural_painters:
      with torch.no_grad():
        decoded = neural_painter(torch.FloatTensor([new_ac]))
      decoded = np.transpose(decoded.numpy(), [0, 2, 3, 1])[0]
      decoded = (decoded * 255).astype(np.uint8)
      stack_these.append(decoded)
    return np.concatenate(stack_these, axis=0)

  clip = mpy.VideoClip(frame, duration=duration)
  clip.write_videofile(video_path, fps=fps)
  print("written to {}".format(video_path))
