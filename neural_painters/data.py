"""Contains utilities for data loading"""
import os
import numpy as np

import torch
from torch.utils.data import RandomSampler, BatchSampler


class ActionStrokeDataLoader:
  """Loads action-stroke pairs from a shard"""
  def __init__(self, path, batch_size, drop_last):
    print('Loading {}'.format(path))
    loaded = np.load(path)
    self.loaded_strokes = loaded['strokes']
    self.loaded_actions = loaded['actions']
    self.sampler = BatchSampler(RandomSampler(self.loaded_strokes),
                                batch_size=batch_size, drop_last=drop_last)

  def __iter__(self):
    for idx in self.sampler:
      # swap color axis because
      # numpy image: B x H x W x C
      # torch image: B x C X H X W
      yield {'stroke': torch.from_numpy(self.loaded_strokes[idx].transpose((0, 3, 1, 2)).astype(np.float)/255.0),
             'action': torch.from_numpy(self.loaded_actions[idx])}


class FullActionStrokeDataLoader:
  """Loads action-stroke pairs from a directory of shards"""
  def __init__(self, dirname, batch_size, drop_last):
    self.dir = dirname
    self.file_list = os.listdir(dirname)
    self.batch_size = batch_size
    self.drop_last = drop_last

  def __iter__(self):
    np.random.shuffle(self.file_list)
    for path in self.file_list:
      loader = ActionStrokeDataLoader(os.path.join(self.dir, path), self.batch_size, self.drop_last)
      for batch in loader:
        yield batch
