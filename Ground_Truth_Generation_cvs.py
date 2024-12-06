import SimpleITK as sitk
import numpy as np
import math
import pandas as pd

def calculate_x_tip_y_tip_angle(T):
    """
    Calculate x_tip, y_tip, and angle based on the transformation matrix T.
    
    Args:
        T (np.ndarray): The 4x4 transformation matrix.
        
    Returns:
        float, float, float: The values of x_tip, y_tip, and angle.
    """
    # Constants Transformation Matrix from Calibratiaon in 3D Slicer
    ProbeToImage = np.array([
        [0.0673051, 0.857019, 0.0963037, 7.35097],
        [0.854817, -0.0535005, -0.121308, -46.6742],
        [-0.114228, 0.104604, -0.851056, 8.71998],
        [0, 0, 0, 1]
    ])

    NeedleTipToNeedle = np.array([
        [-0.0209245, -0.00692098, 0.999757, 99.6817],
        [0.999781, 0, 0.020925, 2.08634],
        [-0.000144821, 0.999976, 0.00691947, 0.689912],
        [0, 0, 0, 1]
    ])
    # Corner points of ultrasoun image in world coordinates
    x_min, y_min = -28.7970008850097, -6.87299919128417
    x_max, y_max = 24.7079975754022, 50.4599991589784
    img_width, img_height = 660, 616

    k_1 = -1 * img_width / (x_max - x_min)
    k_2 = img_height / (y_max - y_min)

    # Calibrate transformation matrix
    T_calibrated = ProbeToImage @ T @ NeedleTipToNeedle

    # Calculated x_tip and y_tip in pixel frame
    x_tip = k_1 * (T_calibrated[0, 3] - x_max)
    y_tip = k_2 * (T_calibrated[1, 3] - y_min)

    # Calculated rotation matrix and normalize
    R_calibrated = T_calibrated[:3, :3]
    u, _, vt = np.linalg.svd(R_calibrated)
    R_norm = u @ vt

    # Calculated angle
    angle = math.atan(R_norm[1, 2] / R_norm[0, 2]) * 180 / math.pi

    return x_tip, y_tip, angle

def load_transform_matrix_from_mha(mha_file_path, slice_index):
    """
    Load the transformation matrix for a specified slice index from an .mha file.
    
    Args:
        mha_file_path (str): Path to the .mha file.
        slice_index (int): Slice index to extract the corresponding transformation matrix.
        
    Returns:
        np.ndarray: The transformation matrix as a 4x4 numpy array, or None if not found.
    """
    # Load the .mha file using SimpleITK
    mha_image = sitk.ReadImage(mha_file_path)
    
    # Create the metadata key based on the slice index
    transform_key = f"Seq_Frame{slice_index:04d}_OTS_Data_0913Transform"
    
    # Check if the key exists in the metadata
    if mha_image.HasMetaDataKey(transform_key):
        # Get the transform matrix values and split them into a list of floats
        matrix_values = list(map(float, mha_image.GetMetaData(transform_key).split()))
        
        # Reshape the list of values into a 4x4 transformation matrix
        transform_matrix = np.array(matrix_values).reshape(4, 4)
        return transform_matrix
    else:
        print(f"Error: No transformation matrix found for slice index {slice_index}.")
        return None

# Main code
mha_file = "./OTS_Data_0913.seq.mha"
start_frame = int(input("Enter the first frame index: "))
end_frame = int(input("Enter the last frame index: "))

# List to store results for CSV
results = []

# Loop over the range of frames
mha_image = sitk.ReadImage(mha_file)  # Load the .mha file once to speed up access

for slice_index in range(start_frame, end_frame + 1):
    transform_matrix = load_transform_matrix_from_mha(mha_file, slice_index)
    if transform_matrix is not None:
        x_tip, y_tip, angle = calculate_x_tip_y_tip_angle(transform_matrix)
        results.append({"Frame": slice_index, "x_tip": x_tip, "y_tip": y_tip, "angle": angle})
    else:
        print(f"No transformation matrix found for frame {slice_index}")

# Convert results to a DataFrame and save as CSV
df = pd.DataFrame(results)
csv_file = "output_x_tip_y_tip_angle.csv"
df.to_csv(csv_file, index=False)

print(f"Results saved to {csv_file}")
