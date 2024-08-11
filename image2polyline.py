from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the image
image_path = 'combined_shapes.png'  
image = Image.open(image_path)

# Convert image to grayscale (optional, but generally binary images are already grayscale)
image = image.convert('L')

# Get image size
width, height = image.size

# Create an empty list to store pixel coordinates
pixel_data = []

# Convert image to numpy array
pixels = np.array(image)

# Define the color to filter (white in this case, value 255)
color_value = 255

for y in range(height):
    for x in range(width):
        if pixels[y, x] ==0:
            # Store x, y coordinates
            pixel_data.append([x, y])

# Convert list to DataFrame
df = pd.DataFrame(pixel_data, columns=['X', 'Y'])

# Save to CSV
csv_path = 'filtered_pixel_data.csv'
df.to_csv(csv_path, index=False)

print(f"Filtered pixel data saved to {csv_path}")

# Plotting
plt.scatter(df['X'], df['Y'], c='black', s=1)  # Using black color for the plotted points
plt.gca().invert_yaxis()  # Invert y axis to match image coordinate system
plt.title('Binary Image Pixel Plot')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
