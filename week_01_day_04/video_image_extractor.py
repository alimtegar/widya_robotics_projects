import cv2

# Load the video
cap = cv2.VideoCapture('parking_cars_footage.mp4')

# Check if the video was opened successfully
if not cap.isOpened():
    print('Error opening video file')
    exit()

# Set the frame number to extract
frame_number = 100

# Set the current frame to the frame number
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the current frame
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print('Error reading frame')
    exit()

# Save the frame as an image
cv2.imwrite('output_image.jpg', frame)

# Release the video
cap.release()