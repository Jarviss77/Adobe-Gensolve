from PIL import Image
import numpy as np
import pandas as pd

def extract_pixel_data(image_path, 
                       csv_path="master_folder\\utils\\output\\out_csv\\filtered_pixel_data.csv"):
    """
    Extracts pixel coordinates from a binary image and saves them to a CSV file.

    Parameters:
        image_path (str): Path to the input image file.
        csv_path (str): Path to the output CSV file.
        color_value (int): Pixel value to filter (default is 255 for white pixels).

    Returns:
        pd.DataFrame: DataFrame containing pixel coordinates.
    """
    # Load the image
    image = Image.open(image_path)

    # Convert image to grayscale (optional, but generally binary images are already grayscale)
    image = image.convert('L')

    # Get image size
    width, height = image.size

    # Create an empty list to store pixel coordinates
    pixel_data = []

    # Convert image to numpy array
    pixels = np.array(image)

    for y in range(height):
        for x in range(width):
            if pixels[y, x] >0 :
                # Store x, y coordinates
                pixel_data.append([0, 0, x, y])

    # Convert list to DataFrame
    df = pd.DataFrame(pixel_data, columns=["curveid", "shape_id", "X", "Y"])

    # Save to CSV
    df.to_csv(csv_path, index=False)

    print(f"Filtered pixel data saved to {csv_path}.")
    return df

# Example usage (uncomment the following lines to test the function):
# image_path = 'master_folder//utils//output//algo4//combined_shapes_occlusion2.png'
# csv_path = 'master_folder\\utils\\output\\out_csv\\filtered_pixel_data.csv'
# extract_pixel_data(image_path, csv_path)
