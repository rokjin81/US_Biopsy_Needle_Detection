import SimpleITK as sitk
import numpy as np
import math
import pandas as pd

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
    transform_key = f"Seq_Frame{slice_index:04d}_OTS_Data_1209Transform"
    
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

def calculate_r_encode_l_encode(T):
    """
    Calculate r_encode and l_encode based on the transformation matrix T.
    
    Args:
        T (np.ndarray): The 4x4 transformation matrix.
        
    Returns:
        float, float: The values of r_encode and l_encode.
    """
    # Constants determined by device dimension
    a = 46
    b = 55
    c = 111

    # Calculate r_encode
    r_encode = 90 - math.atan(T[1, 0] / T[0, 0]) * 180 / math.pi

    # Calculate phi
    phi = math.atan(T[1, 0] / T[0, 0]) + math.atan(b / a)

    # Calculate l_encode
    l_encode = (c - math.sqrt(a**2 + b**2) * math.cos(phi)) - \
               math.sqrt((c - math.sqrt(a**2 + b**2) * math.cos(phi))**2 -
                         a**2 - b**2 - c**2 + 2 * c * math.cos(phi) * math.sqrt(a**2 + b**2) +
                         T[0, 3]**2 + T[1, 3]**2)

    return r_encode, l_encode

# Main code
mha_file = "./OTS_Data_Veri_1209.seq.mha"
start_frame = int(input("Enter the first frame index: "))
end_frame = int(input("Enter the last frame index: "))

# List to store results for CSV
results = []

# Loop over the range of frames
mha_image = sitk.ReadImage(mha_file)  # Load the .mha file once to speed up access

for slice_index in range(start_frame, end_frame + 1):
    transform_matrix = load_transform_matrix_from_mha(mha_file, slice_index)
    if transform_matrix is not None:
        r_encode, l_encode = calculate_r_encode_l_encode(transform_matrix)
        results.append({"Frame": slice_index, "r_encode": r_encode, "l_encode": l_encode})
    else:
        print(f"No transformation matrix found for frame {slice_index}")

# Convert results to a DataFrame and save as CSV
df = pd.DataFrame(results)
csv_file = "output_r_l_encode.csv"
df.to_csv(csv_file, index=False)

print(f"Results saved to {csv_file}")
