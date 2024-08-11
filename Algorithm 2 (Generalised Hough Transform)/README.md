# Generalized Hough Transform Algorithm for Shape Detection (Algorithm 2)
![Output using algo1](../outputs/algo2/tc1.png)

![Output using algo1](../outputs/algo2/tc2.png)

--Output Using Algorithm 2
This algorithm detects shapes in images using the Generalized Hough Transform. It involves several steps including edge detection, gradient orientation computation, and gradient accumulation.

## Key Components and Steps

### 1. **Edge Detection and Gradient Orientation**
   - **Canny Edge Detection:**
     - Uses the `canny` function to detect edges in the image, effective for identifying boundaries.
   - **Gradient Orientation Calculation:**
     - Computes gradient orientation using the Sobel filter, with gradients calculated in both x and y directions (`dx` and `dy`).

### 2. **R-Table Construction**
   - **Building the R-Table:**
     - Constructs an R-Table based on detected edges and their gradient orientations. It stores relative positions (vectors) of edge points with respect to a reference point.

### 3. **Accumulation of Gradients**
   - **Accumulate Gradients Using R-Table:**
     - Uses the R-Table to accumulate votes in an accumulator array. For each edge point, the function looks up corresponding vectors and increments the accumulator.

### 4. **Overlay and Visualization**
   - **Overlay the Reference Image:**
     - The `overlay_reference_image` function overlays the reference image on the query image at the detected position.
   - **Visualization:**
     - The `test_general_hough` function visualizes the results including the reference image, query image with detected points, the accumulator, and the final overlay.

### 5. **Testing and Usage**
   - **Shape-to-Image Conversion:**
     - Converts a set of shapes into binary images using the `shapes_to_image` function.
   - **Testing:**
     - The `test` function reads shape data from CSV files, converts them to images, and applies the Generalized Hough Transform.

## Example Workflow

1. **Read and Convert Shape Data:**
   - Read shape data from CSV files and convert them to binary images.

2. **Apply Generalized Hough Transform:**
   - Apply the Generalized Hough Transform to the query image using the reference images.

3. **Visualize Results:**
   - Visualize the results including the best match and the overlayed reference image.

## Algo 2 Summary

- **Edge Detection:** Uses Canny edge detection and Sobel filter.
- **R-Table Construction:** Creates a lookup table for edge points.
- **Gradient Accumulation:** Accumulates votes in an accumulator array.
- **Overlay and Visualization:** Displays detected shapes and overlays the reference image..

---

