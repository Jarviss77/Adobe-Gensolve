import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(data[:, 0]):
        npXYs = data[data[:, 0] == i][:, 2:]  # Skip the polyline_id and point_id columns
        path_XYs.append([npXYs])
    
    return path_XYs

# Step 2: Convert polylines to an image
def polylines_to_image(path_XYs, width, height):
    img = np.zeros((height, width), dtype=np.uint8)

    for polyline in path_XYs:
        for path in polyline:
            points = path.astype(int)
            for i in range(len(points) - 1):
                cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color=255, thickness=1)

    return img

# Step 3: Apply Canny Edge Detection
def apply_canny(image):
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    return edges

# Step 4: Convert edges to polylines
def edges_to_polylines(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polylines = []

    for contour in contours:
        polyline = contour[:, 0, :]  # Extract x, y coordinates
        polylines.append(polyline)

    return polylines

# Step 5: Save polylines to a new CSV file
def save_polylines_to_csv(polylines, csv_path):
    with open(csv_path, 'w') as f:
        for polyline_id, polyline in enumerate(polylines):
            for point_id, point in enumerate(polyline):
                f.write(f"{polyline_id},{point_id},{point[0]},{point[1]}\n")

# Define file paths and image size
csv_path = 'tc/isolated.csv'
output_csv_path = 'tc/edges_polylines.csv'
width, height = 900,900  # Define the size of your image

# Execute steps
path_XYs = read_csv(csv_path)
image = polylines_to_image(path_XYs, width, height)

# Visualize the original polylines image
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
plt.title('Polylines Image')
plt.axis('off')
plt.show()

edges = apply_canny(image)

# Visualize the edges
plt.figure(figsize=(10, 10))
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')
plt.show()

polylines = edges_to_polylines(edges)
save_polylines_to_csv(polylines, output_csv_path)

output_csv_path
