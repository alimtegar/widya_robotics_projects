import io
import base64
import torch
from fastapi import FastAPI, File, UploadFile
from albumentations import HorizontalFlip, Normalize, Compose
from albumentations.pytorch import ToTensorV2
from skimage import io as skimage_io, transform

from model import UNet
from utils import load_checkpoint, convert_mask

app = FastAPI()

# Configs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = './models/checkpoint'

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

transforms_list = []
transforms_list.extend([HorizontalFlip(p=0.5)])
transforms_list.extend([Normalize(mean=0.5, std=0.5, p=1),
                        ToTensorV2()])
transforms = Compose(transforms_list)

# Load model
model, optimizer, start_epoch, val_loss_min = load_checkpoint(CHECKPOINT_PATH, model, optimizer)

@app.get('/')
def read_root():
    return {'message': 'Hello, world!'}

@app.post('/segment-nuclei')
async def create_segment_nuclei(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    file_stream = io.BytesIO(file_bytes)
    
    image = skimage_io.imread(file_stream)[:, :, :3].astype('float32')
    image = transform.resize(image, (128, 128))
    image = transforms(image=image)['image']
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image)
        
        mask_pred = convert_mask(output[0])
        mask_buffer = io.BytesIO()
        skimage_io.imsave(mask_buffer, mask_pred, format='jpg')
        mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        
        return {'mask': mask_b64}