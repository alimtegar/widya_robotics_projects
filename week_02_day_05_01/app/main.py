import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

app = FastAPI()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = [0.3148, 0.2882, 0.2980]
STD = [0.2944, 0.2631, 0.2215]
CLASSES = ['beverage', 'meat', 'sweet', 'vegetable']
NUM_CLASSES = len(CLASSES)

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv1 = nn.Conv2d(3, 64, 3)
    self.conv2 = nn.Conv2d(64, 128, 3)
    self.fc1 = nn.Linear(128*54*54, 1024)
    self.fc2 = nn.Linear(1024, NUM_CLASSES)

  def forward(self, x):
                                          # -> n, 3, 224, 224
    x = self.pool(F.relu(self.conv1(x)))  # -> n, 64, 111, 111
    x = self.pool(F.relu(self.conv2(x)))  # -> n, 128, 54, 54
    x = x.view(-1, 128*54*54)             # -> n, 373248
    x = F.relu(self.fc1(x))               # -> n, 1024
    x = self.fc2(x)                       # -> n, 4
    return x
  
model = ConvNet().to(DEVICE)
model.load_state_dict(torch.load('./model/custom_cnn_food_dataset_2023-03-04_10-56.pth'))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

@app.get('/')
def read_root():
    return {'message': 'Hello, world!'}

@app.post('/categorize-food')
async def create_categorize_food(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(file.file.read()))
    img = transform(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        print('output', output)
        output_norm = F.softmax(output, dim=1)
        print('output_norm', output_norm)
        pred_prob, pred_class = torch.max(output_norm, dim=1)
    
    return { 'prob': pred_prob.item(), 'class': CLASSES[pred_class], }