import torch


def reconstruction_loss_function(recon_x, x, mask_multiplier):
  """
  This is MSE for images but with non-whitespace areas of the image weighted by mask_multiplier.
  For VAE training, we set this to >1 at the beginning of training for ~10k steps to make sure the VAE does not get
  stuck at predicting pure whitespace.
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
