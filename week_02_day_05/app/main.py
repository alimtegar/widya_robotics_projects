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
    self.conv1 = nn.Conv2d(3, 16, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 3)
    self.fc1 = nn.Linear(32*54*54, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, NUM_CLASSES)

  def forward(self, x):
                                          # -> n, 4, 224, 224
    x = self.pool(F.relu(self.conv1(x)))  # -> n, 32, 111, 111
    x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 54, 54
    x = x.view(-1, 32*54*54)              # -> n, 93321
    x = F.relu(self.fc1(x))               # -> n, 128
    x = F.relu(self.fc2(x))               # -> n, 64
    x = self.fc3(x)                       # -> n, 4
    return x
  
model = ConvNet().to(DEVICE)
model.load_state_dict(torch.load('./model/custom_cnn_food_dataset_2023-03-03_16-34.pth'))

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
    print(img)
    img = transform(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        print(output[0, 0])
        pred = torch.argmax(output, dim=1)
    
    return { 'score': output[0, pred.item()].item(), 'category': CLASSES[pred.item()], }