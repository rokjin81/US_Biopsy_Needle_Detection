import os
import cv2
import torch
import math
import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('./best.pt')  # Path to the trained YOLOv8 .pt model

# Path to the folder containing images and the CSV file
image_folder = './images/'  # Folder containing the usimgXXXXX.png images
csv_file = './output_r_l_encode.csv'

# Load the CSV file
csv_data = pd.read_csv(csv_file)

# Initialize the LSTM input list
lstm_inputs = []

# Loop through all image files in the folder
image_files = sorted([f for f in os.listdir(image_folder) if f.startswith('usimg') and f.endswith('.png')])

for image_file in image_files:
    # Extract frame number from the file name
    frame_number = int(image_file[5:10])

    # Read the image
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)

    # Perform object detection with YOLO
    results = model(frame)

    # Initialize variables for this frame
    bottom_left_x, bottom_left_y, box_angle = -1, -1, -1

    # Check if any objects are detected
    if results and len(results[0].boxes):
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Convert bounding box info to numpy array
            for box in boxes:
                class_id = int(box.cls)  # Class ID of the detected object
                if class_id == 1:  # Assuming class ID of 'needle_us' is 1
                    # Get coordinates of the bounding box (x_min, y_min, x_max, y_max)
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                    # Calculate the bottom-left corner and the angle of the diagonal
                    bottom_left_x, bottom_left_y = x_min, y_max
                    delta_x = x_max - x_min
                    delta_y = y_max - y_min
                    box_angle = math.degrees(math.atan2(delta_y, delta_x))

                    # Only process the first detected object for this frame
                    break

    # Retrieve r_encode and l_encode from the CSV file
    r_encode, l_encode = -1, -1
    csv_row = csv_data[csv_data['Frame'] == frame_number]
    if not csv_row.empty:
        r_encode = csv_row['r_encode'].values[0]
        l_encode = csv_row['l_encode'].values[0]

    # Append the input data to the LSTM input list
    lstm_inputs.append([frame_number, bottom_left_x, bottom_left_y, box_angle, r_encode, l_encode])

# Convert LSTM inputs to a DataFrame and save for verification
lstm_inputs_df = pd.DataFrame(lstm_inputs, columns=['frame_number', 'bottom_left_x', 'bottom_left_y', 'box_angle', 'r_encode', 'l_encode'])
lstm_inputs_df.to_csv('lstm_inputs.csv', index=False)

print("LSTM input data saved to lstm_inputs.csv")