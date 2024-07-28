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
        path_XYs.append([npXYs])
    
    return path_XYs

# Step 2: Convert polylines to an image
def polylines_to_image(path_XYs, width, height):
    img = np.zeros((height, width), dtype=np.uint8)

    for polyline in path_XYs:
        for path in polyline:
            points = path.astype(int)
            for i in range(len(points) - 1):
                cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color=255, thickness=2)

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
    kernel = np.ones((3,3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    plt.figure(figsize=(10, 10))
    plt.imshow(morphed)
    plt.title('morphs')
    plt.axis('off')
    plt.show()
    return morphed

# Step 5: Approximate Contours
def approximate_contours(contours, epsilon_factor=0):
    approx_contours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        print(len(approx))
        if len(approx) < 5:
           approx = cv2.approxPolyDP(cnt, epsilon / 2, True)
        approx_contours.append(approx)
    return approx_contours

# Step 6: Draw Approximated Contours
def draw_approximated_contours(img, approx_contours):
    for cnt in approx_contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            # Draw approximated contours with a custom color (e.g., blue)
            cv2.drawContours(img, [cnt], -1, (255, 0, 0), 7)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title('Approximated Contours')
    plt.axis('off')
    plt.show()

# Define file paths and image size
csv_path = 'tc/frag0.csv'
output_csv_path = 'tc/edges_polylines.csv'
width, height = 512, 512  # Define the size of your image

# Execute steps
path_XYs = read_csv(csv_path)
image = polylines_to_image(path_XYs, width, height)

# Apply Canny Edge Detection and show edges
edges = apply_canny(image)

# Apply morphological operations to separate close objects
morphed_edges = apply_morphology(edges)

# Create a blank color image for drawing
blank_image = np.zeros((height, width, 3), dtype=np.uint8)

# Find contours and approximate them
contours, hierarchy = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
approx_contours = approximate_contours(contours, epsilon_factor=0.02)

# Draw approximated contours on the blank image
draw_approximated_contours(blank_image, approx_contours)

# Optionally save the polylines to CSV
# polylines = edges_to_polylines(edges)
# save_polylines_to_csv(polylines, output_csv_path)
