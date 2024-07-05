# https://github.com/facebookresearch/dinov2?tab=readme-ov-file

import torch
import cv2

# Load the model
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

# Send model to device and set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vitb14_reg.to(device)
dinov2_vitb14_reg.eval()


# Open the video file
video_path = r'C:\Work\Dinov2\Data\ic9LAfKoFTA.001885-001962.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read until the video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Naive resize to 518x518
    frame = cv2.resize(frame, (518, 518))

    # Convert the frame to a tensor
    input_img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Send the input tensor to the device
    input_img = input_img.to(device)

    # Perform inference
    output = dinov2_vitb14_reg(input_img)

    # Save the output tensor to a file
    # torch.save(output, 'output.pt') # Je potreba zmenit

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
