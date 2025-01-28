import SimpleITK as sitk
import numpy as np
import math
import pandas as pd
from tqdm import tqdm  # Progress bar

def calculate_x_tip_y_tip_angle(T, img_width, img_height):
    """
    Calculate x_tip, y_tip, and angle based on the transformation matrix T.
    
    Args:
        T (np.ndarray): The 4x4 transformation matrix.
        img_width (int): Width of the ultrasound image.
        img_height (int): Height of the ultrasound image.
        
    Returns:
        float, float, float: The values of x_tip, y_tip, and angle, or (-1, -1, -1) for invalid cases.
    """
    # Constants Transformation Matrix from Calibration in 3D Slicer
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
    
    # Corner points of ultrasound image in world coordinates
    x_min, y_min = -28.797000885009766, -6.872999191284178
    x_max, y_max = 28.535997465252876, 46.631999269127846

    k_1 = -1 * img_width / (x_max - x_min)
    k_2 = img_height / (y_max - y_min)

    # Calibrate transformation matrix
    T_calibrated = ProbeToImage @ T @ NeedleTipToNeedle

    # Calculate x_tip and y_tip in pixel frame
    x_tip = k_1 * (T_calibrated[0, 3] - x_max)
    y_tip = k_2 * (T_calibrated[1, 3] - y_min)

    # Check for invalid coordinates
    if x_tip < 0 or x_tip > img_width or y_tip < 0 or y_tip > img_height:
        return -1, -1, -1

    # Calculate rotation matrix and normalize
    R_calibrated = T_calibrated[:3, :3]
    u, _, vt = np.linalg.svd(R_calibrated)
    R_norm = u @ vt

    # Calculate angle
    angle = math.atan(R_norm[1, 2] / R_norm[0, 2]) * 180 / math.pi

    return x_tip, y_tip, angle

def load_transform_matrix_from_mha(mha_image, slice_index):
    """
    Load the transformation matrix for a specified slice index from an .mha image.
    
    Args:
        mha_image (SimpleITK.Image): The loaded .mha image.
        slice_index (int): Slice index to extract the corresponding transformation matrix.
        
    Returns:
        np.ndarray: The transformation matrix as a 4x4 numpy array, or None if not found.
    """
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
        print(f"Error: No transformation matrix found for frame {slice_index}.")
        return None

# Main code
mha_file = "./OTS_Data_Veri_1209.seq.mha"
start_frame = int(input("Enter the first frame index: "))
end_frame = int(input("Enter the last frame index: "))
img_width, img_height = 660, 616  # Image dimensions

# List to store results for CSV
results = []

# Load the .mha file once to speed up access
mha_image = sitk.ReadImage(mha_file)

# Loop over the range of frames
for slice_index in tqdm(range(start_frame, end_frame + 1), desc="Processing frames"):
    transform_matrix = load_transform_matrix_from_mha(mha_image, slice_index)
    if transform_matrix is not None:
        x_tip, y_tip, angle = calculate_x_tip_y_tip_angle(transform_matrix, img_width, img_height)
        results.append({"Frame": slice_index, "x_tip": x_tip, "y_tip": y_tip, "angle": angle})

# Convert results to a DataFrame and save as CSV
df = pd.DataFrame(results)
csv_file = "output_x_tip_y_tip_angle.csv"
df.to_csv(csv_file, index=False)

print(f"Results saved to {csv_file}")
