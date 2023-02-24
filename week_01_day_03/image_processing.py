import argparse
import os
import cv2

parser = argparse.ArgumentParser(description='Crop, grayscale, and invert an image.')
parser.add_argument('img_file_path', metavar='img_file_path', type=str, help='path to image file.')
args = parser.parse_args()

img_file_path = args.img_file_path
img_file_name = os.path.splitext(os.path.basename(img_file_path))[0]
img_file_ext = os.path.splitext(img_file_path)[1]

img = cv2.imread(img_file_path)
img_ori = img.copy()

cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
processed_images, processed_image_names = [], []

def save_processed_images(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    for i, _img in enumerate(processed_images):
      cv2.imwrite(f'./{processed_image_names[i]}{img_file_ext}', cv2.cvtColor(_img, cv2.COLOR_RGB2BGR))
    print('Saved!')

def mouse_crop(event, x, y, flags, param):
    # Grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, processed_images, processed_image_names

    # If the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # If the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # Cropping is finished
        
        print('Cropped!')
        
        # Handle crop from every gestures 
        if x_start > x_end:
          x_start, x_end = x_end, x_start
        if y_start > y_end:
          y_start, y_end = y_end, y_start

        # Crop image and 
        img_cropped = img_ori[y_start:y_end, x_start:x_end]
        
        # Resize the cropped image
        # h, w = img_cropped.shape[:2]
        # img_resized = cv2.resize(img_cropped, (int(w / 2), int(h / 2)))
        
        # Process the cropped image
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_neg = cv2.bitwise_not(img_cropped)
        
        processed_images = (img_cropped, img_gray, img_neg)
        processed_image_names = (
          f'{img_file_name}_cropped',
          f'{img_file_name}_gray',
          f'{img_file_name}_neg',
        )
        
        # Concatenate and show the processed images
        img_concated = cv2.hconcat(processed_images)
        cv2.imshow('Processed Image', img_concated)
        
        # Save the processed images
        cv2.setMouseCallback('Processed Image', save_processed_images)

cv2.namedWindow('Original Image')
cv2.setMouseCallback('Original Image', mouse_crop)

while True:
    i = img.copy()

    if not cropping:
        # cv2.line(img, (0, y_move), (img.shape[1], y_move), (0, 255, 0), 2)
        # cv2.line(img, (x_move, 0), (x_move, img.shape[1]), (0, 255, 0), 2)
        cv2.imshow('Original Image', img)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow('Original Image', i)

    cv2.waitKey(1)

# close all open windows
cv2.destroyAllWindows()