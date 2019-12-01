import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.tensorboard import SummaryWriter

from neural_painters.data import FullActionStrokeDataLoader


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# TODO: reconstruction loss
class Discriminator(nn.Module):
  def __init__(self, action_size, dim=16):
    super(Discriminator, self).__init__()
    self.dim = dim

    self.fc1 = nn.Linear(action_size, dim)
    self.conv1 = nn.Conv2d(3, dim, 4, stride=2, padding=1)
    self.conv2 = nn.Conv2d(dim, dim*2, 4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(dim*2)
    self.conv3 = nn.Conv2d(dim*2, dim*4, 4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(dim*4)
    self.conv4 = nn.Conv2d(dim*4, dim*8, 4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(dim*8)
    self.fc2 = nn.Linear(4*4*(dim*8), 1)
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

  def forward(self, images, actions):
    actions = F.relu(self.fc1(actions))
    actions = actions.view(-1, self.dim, 1, 1)
    x = self.leaky_relu(self.conv1(images))

    x = x + actions
    x = self.leaky_relu(self.bn2(self.conv2(x)))
    x = self.leaky_relu(self.bn3(self.conv3(x)))
    x = self.leaky_relu(self.bn4(self.conv4(x)))
    x = x.flatten(start_dim=1)
    x = self.fc2(x)
    return x


class Generator(nn.Module):
  def __init__(self, action_size, dim=16, noise_dim=16, num_deterministic=0):
    super(Generator, self).__init__()
    self.dim = dim
    self.noise_dim = noise_dim
    self.num_deterministic = num_deterministic

    self.fc1 = nn.Linear(action_size + noise_dim, 4*4*(dim*16))  # This seems.. wrong.  Should it be dim*8?
    self.bn1 = nn.BatchNorm2d(dim*16)
    self.deconv1 = nn.ConvTranspose2d(dim*16, dim*8, 4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(dim*8)
    self.deconv2 = nn.ConvTranspose2d(dim*8, dim*4, 4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(dim*4)
    self.deconv3 = nn.ConvTranspose2d(dim*4, dim*2, 4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(dim*2)
    self.deconv4 = nn.ConvTranspose2d(dim*2, 3, 4, stride=2, padding=1)
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

  def forward(self, actions):
    if self.noise_dim > 0:
      batch_size = actions.shape[0]
      noise_concat = torch.randn(batch_size, self.noise_dim - self.num_deterministic).to(actions.device)
      determ_concat = torch.ones(batch_size, self.num_deterministic).to(actions.device) * 0.5
      actions = torch.cat([actions, noise_concat, determ_concat], dim=1)

    x = self.fc1(actions)
    x = x.view(-1, self.dim*16, 4, 4)
    x = F.relu(self.bn1(x))
    x = F.relu(self.bn2(self.deconv1(x)))
    x = F.relu(self.bn3(self.deconv2(x)))
    x = F.relu(self.bn4(self.deconv3(x)))
    x = F.sigmoid(self.deconv4(x))
    return x.view(-1, 3, 64, 64)


class GANNeuralPainter(nn.Module):
  """GAN Neural Painter nn.Module for inference"""
  def __init__(self, action_size, dim=16, noise_dim=16, num_deterministic=0):
    super(GANNeuralPainter, self).__init__()

    self.generator = Generator(action_size, dim, noise_dim, num_deterministic)

  def forward(self, x):
    return self.generator(x)

  def load_from_train_checkpoint(self, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    self.generator.load_state_dict(checkpoint['generator_state_dict'])
    print('Loaded from {}. Batch {}'.format(ckpt_path, checkpoint['batch_idx']))


def save_train_checkpoint(savedir: str,
                          name: str,
                          batch_idx: int,
                          discriminator: Discriminator,
                          generator: Generator,
                          opt_disc,
                          opt_gen):
  os.makedirs(savedir, exist_ok=True)
  obj_to_save = {
    'batch_idx': batch_idx,
    'discriminator_state_dict': discriminator.state_dict(),
    'generator_state_dict': generator.state_dict(),
    'opt_disc_state_dict': opt_disc.state_dict(),
    'opt_gen_state_dict': opt_gen.state_dict()
  }
  torch.save(obj_to_save, os.path.join(savedir, '{}_{}.tar'.format(name, batch_idx)))
  torch.save(obj_to_save, os.path.join(savedir, '{}_latest.tar'.format(name)))
  print('saved {}'.format('{}_{}.tar'.format(name, batch_idx)))


def load_from_latest_checkpoint(savedir: str,
                                name: str,
                                discriminator: Discriminator,
                                generator: Generator,
                                opt_disc,
                                opt_gen):
  latest_path = os.path.join(savedir, '{}_latest.tar'.format(name))
  if not os.path.exists(latest_path):
    print('{} not found. starting training from scratch'.format(latest_path))
    return -1
  checkpoint = torch.load(latest_path)
  discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
  generator.load_state_dict(checkpoint['generator_state_dict'])
  opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
  opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
  print('Loaded from {}. Batch {}'.format(latest_path, checkpoint['batch_idx']))
  return checkpoint['batch_idx']


def reconstruction_loss_function(recon_x, x, mask_multiplier):
  """
  This is MSE but with non-whitespace areas of the brushstroke weighted by mask_multiplier.
  """
  if mask_multiplier != 1.:
    mask = torch.mean(x, dim=1)
    stroke_whitespace = torch.eq(mask, torch.ones_like(mask))
    mask = torch.where(stroke_whitespace, torch.ones_like(mask),
                       torch.ones_like(mask) * mask_multiplier)
    mask = mask.view(-1, 1, 64, 64)
    MSE = torch.sum((recon_x - x) ** 2 * mask, dim=[1, 2, 3])
  else:
    mask = None
    MSE = torch.sum((recon_x - x) ** 2, dim=[1, 2, 3])
  MSE = torch.mean(MSE)
  return MSE, mask


def calc_gradient_penalty(discriminator: nn.Module, real_data: torch.Tensor,
                          fake_data: torch.Tensor, actions: torch.Tensor,
                          device: torch.device, scale: float):
  batch_size = real_data.shape[0]
  #epsilon = torch.rand(batch_size, 1)  # in my tf implementation, same epsilon used for all samples in minibatch
  epsilon = torch.rand(1, 1)
  epsilon = epsilon.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size, 3, 64, 64)
  epsilon = epsilon.to(device)

  interpolates = epsilon * real_data + ((1.0 - epsilon) * fake_data)
  interpolates.requires_grad = True

  disc_interpolates = discriminator(interpolates, actions)
  gradients = autograd.grad(disc_interpolates, interpolates,
                            grad_outputs=torch.ones_like(disc_interpolates),
                            create_graph=True)[0]
  gradients = gradients.view(batch_size, -1)

  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * scale

  return gradient_penalty


def train_gan_neural_painter(action_size: int,
                             dim_size: int,
                             batch_size: int,
                             device: torch.device,
                             data_dir: str,
                             noise_dim: int = 16,
                             disc_iters: int = 5,
                             use_reconstruction_loss: bool = False,
                             stop_reconstruction_loss_after_n_steps: int = 1000000,
                             save_every_n_steps: int = 25000,
                             log_every_n_steps: int = 2000,
                             tensorboard_every_n_steps: int = 100,
                             tensorboard_log_dir: str = 'logdir',
                             save_dir: str = 'gan_train_checkpoints',
                             save_name: str = 'gan_neural_painter'):
  # Initialize data loader
  loader = FullActionStrokeDataLoader(data_dir, batch_size, False)

  # Initialize networks and optimizers
  discriminator = Discriminator(action_size, dim=dim_size).to(device).train()
  generator = Generator(action_size, dim=dim_size, noise_dim=noise_dim,
                        num_deterministic=0).to(device).train()  # Must always train fully stochastically
  discriminator.apply(weights_init)
  generator.apply(weights_init)

  optim_disc = optim.Adam(discriminator.parameters(), lr=1e-4)
  optim_gen = optim.Adam(generator.parameters(), lr=1e-4)

  # Initialize networks from latest checkpoint if it exists.
  batch_idx_offset = 1 + load_from_latest_checkpoint(
    save_dir,
    save_name,
    discriminator,
    generator,
    optim_disc,
    optim_gen
  )
  # Initialize tensorboard a.k.a. greatest thing since sliced bread
  writer = SummaryWriter(tensorboard_log_dir)
  for _ in range(100):
    for batch_idx, batch in enumerate(loader):
      batch_idx += batch_idx_offset

      strokes = batch['stroke'].float().to(device)
      actions = batch['action'].float().to(device)

      if (batch_idx + 1) % (disc_iters + 1) == 0:  # Generator step every disc_iters+1 steps
        for p in discriminator.parameters():
          p.requires_grad = False  # to avoid computation (i copied this code, but this makes no sense i think?)
        optim_gen.zero_grad()

        generated = generator(actions)
        generated_score = torch.mean(discriminator(generated, actions))

        generator_loss = generated_score

        if use_reconstruction_loss and batch_idx < stop_reconstruction_loss_after_n_steps:
          uneven_mult = 10. if batch_idx < 100000 else 1.  # Do I need this?
          if batch_idx < 500000:
            reconstruction_loss_mult = 10.
          elif batch_idx < 550000:
            reconstruction_loss_mult = 1.
          else:
            reconstruction_loss_mult = 0.1
          reconstruction_loss, _ = reconstruction_loss_function(generated, strokes, uneven_mult)
          generator_loss += reconstruction_loss_mult*reconstruction_loss
          writer.add_scalar('reconstruction_loss', reconstruction_loss, batch_idx)
          writer.add_scalar('reconstruction_loss_mult', reconstruction_loss_mult, batch_idx)

        generator_loss.backward()
        optim_gen.step()

        writer.add_scalar('generator_loss', generator_loss, batch_idx)

      else:  # Discriminator steps for everything else
        for p in discriminator.parameters():
          p.requires_grad = True  # they are set to False in generator update
        optim_disc.zero_grad()

        real_score = torch.mean(discriminator(strokes, actions))

        generated = generator(actions)
        generated_score = torch.mean(discriminator(generated, actions))

        gradient_penalty = calc_gradient_penalty(discriminator, strokes.detach(),
                                                 generated.detach(), actions,
                                                 device, 10.0)

        disc_loss = real_score - generated_score + gradient_penalty
        disc_loss.backward()
        optim_disc.step()

        writer.add_scalar('discriminator_loss', disc_loss, batch_idx)
        writer.add_scalar('real_score', real_score, batch_idx)
        writer.add_scalar('generated_score', generated_score, batch_idx)
        writer.add_scalar('gradient_penalty', gradient_penalty, batch_idx)

      if batch_idx % tensorboard_every_n_steps == 0:
        writer.add_images('img_in', strokes[:3], batch_idx)
        writer.add_images('img_out', generated[:3], batch_idx)
      if batch_idx % log_every_n_steps == 0:
        print('train batch {}'.format(batch_idx))

      if batch_idx % save_every_n_steps == 0:
        save_train_checkpoint(save_dir, save_name, batch_idx,
                              discriminator, generator,
                              optim_disc, optim_gen)
    batch_idx_offset = batch_idx + 1
