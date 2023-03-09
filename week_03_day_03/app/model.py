import torchvision
from typing import Union, Dict
from torch import nn
from quaterion_models.encoders import Encoder

from encoder import FruitsEncoder

class Model(nn.Module):
  def __init__(self):
    super().__init__()

  def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
    pretrained_encoder = torchvision.models.resnet152(weights=None)
    pretrained_encoder.fc = nn.Identity()
    return FruitsEncoder(pretrained_encoder)