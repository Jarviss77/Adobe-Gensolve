# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:41:37 2024
@author: OM
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(data[:, 0]):
        npXYs = data[data[:, 0] == i][:, 2:]  # Skip the polyline_id and point_id columns
        path_XYs.append(npXYs)
    
    return path_XYs

# Step 2: Convert polylines to an image
# Step 2: Convert polylines to an image
def polylines_to_image(path_XYs, width, height):
    img = np.zeros((height, width), dtype=np.uint8)
    for polyline in path_XYs:
        for path in polyline:
            points = path.astype(int)
            # Check if points is 2D
            if points.ndim == 2:
                for i in range(len(points) - 1):
                    start_point = tuple(points[i])
                    end_point = tuple(points[i + 1])
                    cv2.line(img, start_point, end_point, color=255, thickness=2)
            else:
                raise ValueError("The 'path' array is not 2D.")
    return img



# Step 3: Apply Canny Edge Detection and show edges
def apply_canny(image):
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    plt.figure(figsize=(10, 10))
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    plt.show()
    return edges

# Step 4: Apply morphological operations to separate close objects
def apply_morphology(edges):
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    plt.figure(figsize=(10, 10))
    plt.imshow(morphed, cmap='gray')
    plt.title('Morphological Operation')
    plt.axis('off')
    plt.show()
    return morphed

# Step 5: Approximate Contours
def approximate_contours(contours, epsilon_factor=0.02):
    approx_contours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
        print(f'Contour Circularity: {circularity}')
        if circularity > 0.8:  
            approx = cv2.approxPolyDP(cnt, 0.001 * perimeter, True)  # Small epsilon to keep contour
        approx_contours.append(approx)
    return approx_contours

# Step 6: Draw Approximated Contours
def draw_approximated_contours(img, approx_contours):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in approx_contours:
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.drawContours(color_img, [cnt], -1, color, 3)

    plt.figure(figsize=(10, 10))
    plt.imshow(color_img)
    plt.title('Approximated Contours')
    plt.axis('off')
    plt.show()

# Define file paths and image size
csv_path = 'tc/isolated.csv'
width, height = 512, 512  # Define the size of your image

# Process the CSV file and generate the image
path_XYs = read_csv(csv_path)
image = polylines_to_image(path_XYs, width, height)

# Apply Canny edge detection
edges = apply_canny(image)

# Apply morphological operations
morphed_edges = apply_morphology(edges)

# Find contours and approximate them
contours, hierarchy = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
approx_contours = approximate_contours(contours)

# Draw approximated contours on the image
draw_approximated_contours(morphed_edges, approx_contours)
