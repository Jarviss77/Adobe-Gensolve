import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the CSV file and handle shapes inside shapes
def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    shapes = []

    for i in np.unique(data[:, 0]):
        shape_points = data[data[:, 0] == i][:, 2:]  # Skip the polyline_id and point_id columns
        shapes.append(shape_points)

    return shapes

# Step 2: Convert shapes to individual images
def shapes_to_images(shapes, width, height):
    images = []
    for shape in shapes:
        img = np.zeros((height, width), dtype=np.uint8)
        points = shape.astype(int)
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color=255, thickness=2)
        images.append(img)
    return images

# Step 3: Apply Canny Edge Detection and show edges
def apply_canny(image):
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    return edges

# Step 4: Apply morphological operations to separate close objects
def apply_morphology(edges):
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return morphed

# Step 5: Approximate Contours
def approximate_contours(contours, epsilon_factor=0.04):
    approx_contours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        print(circularity)
        if(circularity>0.043):
            approx = cv2.approxPolyDP(cnt, 0, True)
        approx_contours.append(approx)
    return approx_contours

# Step 6: Draw Approximated Contours
def draw_approximated_contours(img, approx_contours):
    for cnt in approx_contours:
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.drawContours(img, [cnt], -1, color, 2)

# Define file paths and image size
csv_path = 'trash/tc/frag1.csv'
width, height = 512, 512  # Define the size of your image

# Execute steps
shapes = read_csv(csv_path)
shape_images = shapes_to_images(shapes, width, height)

# Create a blank color image for drawing all shapes
combined_blank_image = np.zeros((height, width, 3), dtype=np.uint8)

# Process each shape individually
for shape_img in shape_images:
    # Apply Canny Edge Detection and show edges
    edges = apply_canny(shape_img)

    # Apply morphological operations to separate close objects
    morphed_edges = apply_morphology(edges)

    # Find contours and approximate them
    contours, hierarchy = cv2.findContours(morphed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    approx_contours = approximate_contours(contours)

    # Draw approximated contours on the combined blank image
    draw_approximated_contours(combined_blank_image, approx_contours)
    
plt.figure(figsize=(10, 10))
plt.imshow(combined_blank_image)
plt.title('Approximated Contours')
plt.axis('off')
plt.show()

# Optionally save the polylines to CSV
# polylines = edges_to_polylines(edges)