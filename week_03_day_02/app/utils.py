import numpy as np
import torch

def load_checkpoint(checkpoint_path, model, optimizer):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  val_loss_min = checkpoint['val_loss_min']
  return model, optimizer, checkpoint['epoch'], val_loss_min.item()
  
def convert_mask(mask):
  std = np.array((0.5))
  mean = np.array((0.5))

  mask = mask.clone().cpu().detach().numpy()
  mask = mask.transpose((1, 2, 0))
  mask = std * mask + mean
  mask = mask.clip(0, 1)
  mask = np.squeeze(mask)
  return mask