import os
import json
import numpy as np
from typing import List
from torchvision import transforms

from model import Model

def create_transforms(input_size: int, mean: List[float], std: List[float]):
  transforms_list = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
  ])
  return transforms_list

def load_model(checkpoint_path):
  model = Model()
  model = model.configure_encoders()
  return model.load(checkpoint_path)

def register_embedding(label: str, embedding: np.array):
  if not os.path.exists('db.json'):
    db = {}
  else:
    with open('db.json', 'r') as f:
      db = json.load(f)
  
  db[label] = embedding.tolist()
  
  with open('db.json', 'w') as f:
      json.dump(db, f)
      
def get_cos_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
      
def recognize_embedding(embedding: np.array, threshold: float=0.5):
  if not os.path.exists('db.json'):
    return None
  
  with open('db.json', 'r') as f:
      db = json.load(f)
  
  label_pred = None
  max_similarity = 0
  for label, registered_embedding in db.items():
    similarity = get_cos_similarity(embedding, np.array(registered_embedding))
    if similarity > max_similarity and similarity > threshold:
      max_similarity = similarity
      label_pred = label
    
  return label_pred