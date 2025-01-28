# Needle Detection and Tracking Algorithm for Ultrasound Biopsy using Multimodal  AI
This project proposes a precise needle insertion assistant device and a multimodal artificial intelligence needle detection and tracking algorithm for ultrasound-guided thyroid biopsy procedures. The needle detection algorithm is a multimodal recurrent network combining YOLO (You Only Look Once) v8, which detects needles based on ultrasound images, with LSTM (Long Short-Term Memory) to process the time series data of the 2-DOF motion information of the device. 

## Description

1. **Extract_Image_from_NRRD_withRotation.py**  
   Extracts image data from an NRRD file containing ultrasound image information recorded in 3D Slicer.  
   The output images include the frame number appended to the file name.  
   Specify the `file_name` as the path and file name of the NRRD file.

2. **best.pt**  
   A YOLOv8 model trained for needle detection based on image data.  
   `class ID = 1` corresponds to `needle_us`.

3. **Sythetic_Encoder_Generation_csv.py**  
   Generates synthetic encoder data for the assistive device from an MHA file containing OTS data recorded in 3D Slicer, and saves it as a CSV file.  
   - Specify the `mha_file` as the path and file name of the MHA file.
   - Specify the 'transform_key' as the structure of the MHA file.

5. **Input_Generation_LSTM.py**  
   Combines image processing results from the YOLOv8 model with encoder data from the assistive device to generate input data for the LSTM model, and saves it as a CSV file.  
   - Specify the `model` as the path and file name of the YOLO model.  
   - Specify the `image_folder` as the path to the ultrasound images.  
   - Specify the `csv_file` as the path and file name of the encoder data.

6. **Ground_Truth_Generation_csv.py**  
   Generates ground truth data for needle position and angle from an MHA file containing OTS data recorded in 3D Slicer, and saves it as a CSV file.  
   - Input the results of pivot calibration and pointer calibration from 3D Slicer into the constants `ProbeToImage` and `NeedleTipToNeedle` transformation matrices in the "Constants Transformation Matrix from Calibration in 3D Slicer".  
   - Input the coordinate information of the aligned ultrasound image in 3D Slicer space into the "Corner points of ultrasound image in world coordinates".  
   - Specify the `mha_file` as the path and file name of the MHA file.
   - Specify the 'transform_key' as the structure of the MHA file.

7. **LSTM_training.py**  
   Trains the LSTM model using time-series data combining image data processed through YOLO and encoder values from the assistive device.  
   Specify the training and validation datasets in `Load datasets`.

8. **lstm_model_best.pth**  
   An LSTM model trained using experimental data.

9. **LSTM_output.py**  
   Outputs the needle position and angle based on the trained LSTM model.  
   - Specify the `input_csv` as the path and file name of the CSV file containing the time-series input data.  
   - Specify the `model_path` as the path and file name of the trained LSTM model.  
   - Specify the `input_folder` as the path to the folder containing the input ultrasound images.  

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software for personal or commercial purposes, provided that proper attribution is given to the original authors.

## Contact
Maintainer - Sangrok Jin  
Email - rokjin.kor@gmail.com
