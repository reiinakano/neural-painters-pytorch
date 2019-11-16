import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torchvision import transforms, utils


class VAEEncoder(nn.Module):
  def __init__(self, z_size):
    super(VAEEncoder, self).__init__()

    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc_mu = nn.Linear(2 * 2 * 256, z_size)
    self.fc_log_var = nn.Linear(2 * 2 * 256, z_size)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = x.view(-1, 2 * 2 * 256)
    mu = self.fc_mu(x)
    log_var = self.fc_log_var(x)

    # reparameterization
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std

    return z, mu, log_var


class VAEDecoder(nn.Module):
  def __init__(self, z_size):
    super(VAEDecoder, self).__init__()

    self.fc = nn.Linear(z_size, 4 * 256)
    self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
    self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  def forward(self, x):
    x = self.fc(x)  # No activation?
    x = x.view(-1, 4 * 256, 1, 1)
    x = F.relu(self.deconv1(x))
    x = F.relu(self.deconv2(x))
    x = F.relu(self.deconv3(x))
    x = torch.sigmoid(self.deconv4(x))
    return x


class VAEPredictor(nn.Module):
  def __init__(self, action_size, z_size):
    super(VAEPredictor, self).__init__()

    self.fc1 = nn.Linear(action_size, 256)
    self.bn1 = nn.BatchNorm1d(256)
    self.fc2 = nn.Linear(256, 64)
    self.bn2 = nn.BatchNorm1d(64)
    self.fc3 = nn.Linear(64, 64)
    self.bn3 = nn.BatchNorm1d(64)
    self.fc_mu = nn.Linear(64, z_size)
    self.fc_log_var = nn.Linear(64, z_size)

  def forward(self, x):
    x = self.bn1(F.leaky_relu(self.fc1(x)))
    x = self.bn2(F.leaky_relu(self.fc2(x)))
    x = self.bn3(F.leaky_relu(self.fc3(x)))
    mu = self.fc_mu(x)
    log_var = self.fc_log_var(x)

    # reparameterization
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std

    return z, mu, log_var


class VAENeuralPainter(nn.Module):
  """VAE Neural Painter nn.Module for inference"""
  def __init__(self, action_size, z_size, stochastic=True):
    super(VAENeuralPainter, self).__init__()

    self.stochastic = stochastic
    self.predictor = VAEPredictor(action_size, z_size)
    self.decoder = VAEDecoder(z_size)

  def forward(self, x):
    z, mu, log_var = self.predictor(x)
    if self.stochastic:
      return self.decoder(z)
    else:
      return self.decoder(mu)

  def load_from_train_checkpoint(self, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
    print('Loaded from {}. Batch {}'.format(ckpt_path, checkpoint['batch_idx']))
