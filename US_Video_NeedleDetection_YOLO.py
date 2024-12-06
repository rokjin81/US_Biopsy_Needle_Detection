import cv2
import torch
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('./best.pt')  # Path of the trained YOLOv8 .pt model

# Load MP4 video file
video_path = './output_video.mp4'  # Path of the input video file
output_video_path = 'detection_video.mp4'  # Path of the saving MP4 file

# Load video
cap = cv2.VideoCapture(video_path)

# Get parameters of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Save the video setting (MP4 type, codec: mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Get class ID corresponding to index of "needle_us"
needle_us_index = 1  # For example, class ID of 'needle_us' is '1'.

# Control video frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection from the frame using YOLO model
    results = model(frame)  # Detect with the model
    
    for result in results:
        boxes = result.boxes.cpu().numpy()  # Convert the information of detected bounidng box to NumPy array
        for box in boxes:
            class_id = int(box.cls)  # Class ID of the detected object
            if class_id == needle_us_index:  # If the class is 'needle_us',
                # Get coordinates of the bounding box (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                # Extract the coordinates of left bottom corner as the tip position
                bottom_left = (x_min, y_max)
                cv2.putText(frame, f"({bottom_left[0]}, {bottom_left[1]})", (x_min, y_max + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Calculate the angle of diagonal line of the bounding box
                delta_x = x_max - x_min
                delta_y = y_max - y_min
                angle_rad = math.atan2(delta_y, delta_x)  # Calculate the angle (radians)
                angle_deg = math.degrees(angle_rad)  # Convert to degree

                # Print the angle on the frame
                cv2.putText(frame, f"Angle: {angle_deg:.2f} deg", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save the frame as video
    out.write(frame)

    # quit with 'q' key
    cv2.imshow('YOLOv8 Needle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  
cv2.destroyAllWindows()
