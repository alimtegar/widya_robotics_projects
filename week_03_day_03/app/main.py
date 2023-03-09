import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from utils import create_transforms, load_model, register_embedding, recognize_embedding

app = FastAPI()

# Configs
CHECKPOINT_PATH = './encoders/default'
INPUT_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

model = load_model(checkpoint_path=CHECKPOINT_PATH)
transforms = create_transforms(input_size=INPUT_SIZE, mean=MEAN, std=STD)

@app.get('/')
def read_root():
    return {'message': 'Hello, world!'}
  
@app.post('/register')
async def create_register(label: str, file: UploadFile = File(...)):
  file_bytes = await file.read()
  file_stream = io.BytesIO(file_bytes)
  
  image = Image.open(file_stream).convert('RGB')
  image = transforms(image)
  image = image.unsqueeze(0)
  
  embedding = model(image)
  register_embedding(label, embedding.detach().squeeze().numpy())
  
  return {'Message': 'Successfully registered.'}

@app.post('/recognize')
async def create_recognize(file: UploadFile = File(...), threshold: float=0.5):
  file_bytes = await file.read()
  file_stream = io.BytesIO(file_bytes)
  
  image = Image.open(file_stream).convert('RGB')
  image = transforms(image)
  image = image.unsqueeze(0)
  
  embedding = model(image)
  label_pred = recognize_embedding(embedding.detach().squeeze().numpy(), threshold=threshold)
  
  return {'label': label_pred}

