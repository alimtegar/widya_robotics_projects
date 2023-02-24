import cv2
import numpy as np
import json

img = cv2.imread('parking_space.webp')

with open('parking_space.json') as f:
  mask = json.load(f)
  
pts_list = [mask_shape['points'] for mask_shape in mask['shapes']]

def check_parking_space(img_processed):  
  for pts in pts_list:
    _pts = np.array([pts], dtype=np.int32)

    mask = np.zeros_like(img_processed)
    cv2.fillPoly(mask, _pts, (255, 255, 255))
    img_cropped = cv2.bitwise_and(img_processed, mask)

    px_count = cv2.countNonZero(img_cropped)
    
    if px_count > 2000:
      color = (0, 0, 255)
    else:
      color = (0, 255, 0)
    
    cv2.polylines(img, _pts, True, color, 2)
    # print(px_count)
    
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 1)
img_thres = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
img_median = cv2.medianBlur(img_thres, 5)
kernel = np.ones((3, 3), np.uint8)
img_dilated = cv2.dilate(img_median, kernel, iterations=1)

check_parking_space(img_dilated)

# Show the image
while True:
  cv2.imshow('Image', img)
  cv2.waitKey(1)