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

# Step 3: Apply Canny Edge Detection
def apply_canny(image):
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    return edges

# Step 4: Apply morphological operations to separate close objects
def apply_morphology(edges):
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return morphed

# Step 5: Approximate Contours
def approximate_contours(contours, epsilon_factor=0.03):
    approx_contours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_contours.append(approx)
    return approx_contours

# Step 6: Draw Approximated Contours
def draw_approximated_contours(img, approx_contours):
    for cnt in approx_contours:
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.drawContours(img, [cnt], -1, color, 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title('Approximated Contours')
    plt.axis('off')
    plt.show()

# Step 7: Shape Completion for Connected Occlusion
def complete_connected_occlusion(edges):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = approximate_contours(contours)

    # Connect occluded parts
    completed_edges = np.zeros_like(edges)
    for cnt in approx_contours:
        for i in range(len(cnt) - 1):
            cv2.line(completed_edges, tuple(cnt[i][0]), tuple(cnt[i + 1][0]), 255, thickness=1)
    return completed_edges

# Step 8: Shape Completion for Disconnected Occlusion
def complete_disconnected_occlusion(edges):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = approximate_contours(contours)

    # Connect fragments
    completed_edges = np.zeros_like(edges)
    for cnt in approx_contours:
        for i in range(len(cnt) - 1):
            cv2.line(completed_edges, tuple(cnt[i][0]), tuple(cnt[i + 1][0]), 255, thickness=1)
        # Close the contour
        if len(cnt) > 2:
            cv2.line(completed_edges, tuple(cnt[-1][0]), tuple(cnt[0][0]), 255, thickness=1)
    return completed_edges

# Step 9: Plot Completed Shapes
def plot_completed_shapes(image, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Define file paths and image size
csv_path = 'tc/occlusion1.csv'
width, height = 512, 512  # Define the size of your image

# Execute steps
shapes = read_csv(csv_path)
shape_images = shapes_to_images(shapes, width, height)

# Process each shape individually
for shape_img in shape_images:
    # Apply Canny Edge Detection and show edges
    edges = apply_canny(shape_img)
    
    # Apply morphological operations to separate close objects
    morphed_edges = apply_morphology(edges)

    # Complete shapes for connected occlusion
    completed_connected = complete_connected_occlusion(morphed_edges)
    plot_completed_shapes(completed_connected, 'Connected Occlusion Completed Shapes')

    # Complete shapes for disconnected occlusion
    completed_disconnected = complete_disconnected_occlusion(morphed_edges)
    plot_completed_shapes(completed_disconnected, 'Disconnected Occlusion Completed Shapes')
