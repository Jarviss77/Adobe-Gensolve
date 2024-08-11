# Generalized Hough Transform Algorithm for Shape Detection With Multi-Scale and Multi-Shift Detection (Algorithm 3)
![Output using algo1](../outputs/algo3/tc1.png)

![Output using algo1](../outputs/algo3/tc2.png)

--Output Using Algorithm 3

This advanced algorithm detects shapes using the Generalized Hough Transform with added multi-scale and multi-shift detection capabilities.

## Key Components and Steps

### 1. **Edge Detection and Gradient Orientation**
   - **Canny Edge Detection:**
     - Uses the `canny` function to detect edges effectively.
   - **Gradient Orientation Calculation:**
     - Computes gradient orientation using the Sobel filter.

### 2. **R-Table Construction**
   - **Building the R-Table:**
     - Constructs an R-Table based on detected edges and orientations.

### 3. **Accumulation of Gradients**
   - **Accumulate Gradients Using R-Table:**
     - Accumulates votes in an accumulator array.

### 4. **Multi-Scale and Multi-Shift Detection**
   - **Generate Shifts:**
     - Creates a list of possible shifts for the reference image.
   - **Multi-Scale Detection:**
     - Tests different scales of the reference image to detect size variations.
   - **Multi-Shift Detection:**
     - Tests various shifts to find the best alignment.

### 5. **Overlay and Visualization**
   - **Overlay the Reference Image:**
     - Overlays the reference image on the query image at the detected position.
   - **Visualization:**
     - Visualizes the results including the reference image, query image with detected points, and the final overlay.

### 6. **Testing and Usage**
   - **Shape-to-Image Conversion:**
     - Converts shapes to binary images.
   - **Testing:**
     - Reads shape data from CSV files, converts them to images, and applies the Generalized Hough Transform.

## Example Workflow

1. **Read and Convert Shape Data:**
   - Read and convert shape data from CSV files to binary images.

2. **Apply Generalized Hough Transform:**
   - Apply the Generalized Hough Transform with multi-scale and multi-shift detection.

3. **Visualize Results:**
   - Visualize the best match and overlayed reference image.

## Algo 3 Summary

- **Edge Detection:** Uses Canny edge detection and Sobel filter.
- **R-Table Construction:** Creates a lookup table for edge points.
- **Gradient Accumulation:** Accumulates votes in an accumulator array.
- **Multi-Scale and Multi-Shift Detection:** Tests different scales and shifts.
- **Overlay and Visualization:** Displays the detected shapes and overlays the reference image.
.

---
