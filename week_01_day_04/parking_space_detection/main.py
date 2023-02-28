import cv2
import numpy as np
import json

# Video feed
cap = cv2.VideoCapture('parking_cars_footage_sm.mp4')

with open('parking_cars_footage.json') as f:
  mask = json.load(f)
  
pos_list = [mask_shape['points'] for mask_shape in mask['shapes']]

def check_parking_space(img_processed):  
  colors = { 'red': (0, 0, 255), 'green': (0, 255, 0) }
  space_count = 0
  
  for pos in pos_list:
    _pos = np.array([pos], dtype=np.int32)

    # Crop image based on bounding box coordinates
    mask = np.zeros_like(img_processed)
    cv2.fillPoly(mask, _pos, (255, 255, 255))
    img_cropped = cv2.bitwise_and(img_processed, mask)

    # Count non-zero pixels
    px_count = cv2.countNonZero(img_cropped)
    
    if px_count > 320:
      # If space is occupied
      color = colors['red']
    else:
      # If space is free
      color = colors['green']
      space_count += 1
    
    # Draw bounding box and show pixel count
    cv2.polylines(img, _pos, True, color, 2)
    w, h, p = 20, 10, 2
    cv2.rectangle(img, (_pos[0][0][0], _pos[0][0][1] - h), (_pos[0][0][0] + w, _pos[0][0][1]), color, -1)
    cv2.putText(img, text=str(px_count), org=(_pos[0][0][0] + p, _pos[0][0][1] - h//2 + p), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(0, 0, 0), thickness=1)
  
  # Show free and occupied space count
  y, w, h, p = 10, 169, 24, 5
  x1 = 10
  x2 = x1 + w + 10
  
  cv2.rectangle(img, (x1, y), (x1 + w, y + h), colors['green'], -1)
  cv2.putText(img, text=f'Free Space: {space_count}/{len(pos_list)}', org=(x1 + p, y + h//2 + p), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
  
  cv2.rectangle(img, (x2, y), (x2 + w, y + h), colors['red'], -1)
  cv2.putText(img, text=f'Occupied Space: {len(pos_list)- space_count}', org=(x2 + p, y + h//2 + p), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)

while True:
  retval, img = cap.read()
  
  # Loop video if there's no frame grabbed
  if not retval:
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      continue
  
  # Process image
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 1)
  img_thres = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
  img_median = cv2.medianBlur(img_thres, 5)
  kernel = np.ones((3, 3), np.uint8)
  img_dilated = cv2.dilate(img_median, kernel, iterations=1)
  
  check_parking_space(img_dilated)
  
  cv2.imshow('Image', img)
  cv2.waitKey(10)