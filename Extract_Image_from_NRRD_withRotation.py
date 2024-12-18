import numpy as np
import nrrd
from PIL import Image
import os

filename = "./US_Image_Veri_1209.seq.nrrd"  # Path and filename of NRRD file
readdata, header = nrrd.read(filename)
print(readdata.shape)

# Set saving path as absolute path
save_dir = os.path.join(os.getcwd(), 'png')
os.makedirs(save_dir, exist_ok=True)

# Repeat for number of slides
for i in range(readdata.shape[0]):  
    # Access to each slide
    b = readdata[i, :, :, :]  # (660, 616, 1) Original data dimension
    b = np.squeeze(b)  # Compression of dimension to (660, 616) (Mono channel)
    
    # Convert to uint8 (Pillow requires this type.)
    b = b.astype(np.uint8)
    
    # Rotate image -90 degrees (clockwise)
    final = Image.fromarray(b)
    rotated_img = final.rotate(-90, expand=True)  # Expand ensures the rotated image fits
    
    # Print the path of saved image
    save_path = os.path.join(save_dir, f'usimg{i}.png')
    print(f"Saving rotated slice {i} as PNG: {save_path}")
    
    # Save the rotated image
    rotated_img.save(save_path)

print("All slices have been processed, rotated, and saved.")
