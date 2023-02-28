from typing import List
from enum import Enum
from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
import base64

app = FastAPI()

class Mode(str, Enum):
    grayscale = 'grayscale'
    negative = 'negative'

@app.get('/')
def read_root():
    return {'message': 'Hello, world!'}

@app.post('/process-images')
async def create_process_images(mode: Mode, files: List[UploadFile] = File(...)):
    images = []
    
    # Handling errors
    # Check if mode is valid
    if not mode in ['grayscale', 'negative']:
        raise HTTPException(status_code=400, detail="Invalid mode option. Only 'grayscale' and 'negative' are allowed.")
    
    # Check if number of files is allowed.
    if len(files) > 5:
        raise HTTPException(status_code=400, detail='The maximum number of allowed files (5) has been exceeded.')
    
    # Check if file type is valid
    for file in files:
                          # image/jpeg, image/png
        if not 'image' in file.content_type:
            raise HTTPException(status_code=415, detail='Only image files are allowed for upload.')
    
    # Process images
    for file in files:
        contents = await file.read()
        np_arr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if (mode == 'grayscale'):
            # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif (mode == 'negative'):
            # Invert image to make negative image
            image = cv2.bitwise_not(image)       
        
        _, image_buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(image_buffer).decode('utf-8')

        images.append(image_b64)        
    
    return {
        'mode': mode,
        'images': images,
    }