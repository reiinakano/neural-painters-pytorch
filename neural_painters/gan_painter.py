import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from neural_painters.data import FullActionStrokeDataLoader


class Discriminator(nn.Module):
  def __init__(self, action_size, dim=16):
    super(Discriminator, self).__init__()

    self.fc1 = nn.Linear(action_size, dim)
    self.conv1 = nn.Conv2d(3, dim, 4, stride=2)  # Padding?
    self.conv2 = nn.Conv2d(dim, dim*2, 4, stride=2)
    self.bn2 = nn.BatchNorm2d(dim*2)
    self.conv3 = nn.Conv2d(dim*2, dim*4, 4, stride=2)
    self.bn3 = nn.BatchNorm2d(dim*4)
    self.conv4 = nn.Conv2d(dim*4, dim*8, 4, stride=2)
    self.bn4 = nn.BatchNorm2d(dim*8)
    self.fc2 = nn.Linear(4*4*(dim*8), 1)
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

  def forward(self, images, actions):
    actions = F.relu(self.fc1(actions))
    x = self.leaky_relu(self.conv1(images))

    x = x + actions
    x = self.leaky_relu(self.bn2(self.conv2(x)))
    x = self.leaky_relu(self.bn3(self.conv3(x)))
    x = self.leaky_relu(self.bn4(self.conv4(x)))
    x = x.flatten()
    x = self.fc2(x)
    return x


class Generator(nn.Module):
  def __init__(self, action_size, dim=16):
    super(Generator, self).__init__()
    self.dim = dim

    self.fc1 = nn.Linear(action_size, 4*4*(dim*16))  # This seems.. wrong.  Should it be dim*8?
    self.bn1 = nn.BatchNorm2d(dim*16)
    self.deconv1 = nn.ConvTranspose2d(dim*16, dim*8, 4, stride=2)
    self.bn2 = nn.BatchNorm2d(dim*8)
    self.deconv2 = nn.ConvTranspose2d(dim*8, dim*4, 4, stride=2)
    self.bn3 = nn.BatchNorm2d(dim*4)
    self.deconv3 = nn.ConvTranspose2d(dim*4, dim*2, 4, stride=2)
    self.bn4 = nn.BatchNorm2d(dim*2)
    self.deconv4 = nn.ConvTranspose2d(dim*2, 3, 4, stride=2)
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

  def forward(self, actions):
    # TODO: Add random noise

    x = self.fc1(actions)
    x = x.view(-1, self.dim*16, 4, 4)
    x = F.relu(self.bn1(x))
    x = F.relu(self.bn2(self.deconv1(x)))
    x = F.relu(self.bn3(self.deconv2(x)))
    x = F.relu(self.bn4(self.deconv3(x)))
    x = F.sigmoid(self.deconv4(x))
    return x.view(-1, 3, 64, 64)
