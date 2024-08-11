# Symmetry Detection and Analysis Algorithm (Algorithm 1)
![Output using algo1](../outputs/algo1/tc1.png)

![Output using algo1](../outputs/algo1/tc2.png)

--Output Using Algorithm 1
This algorithm efficiently identifies and analyzes symmetry in images through a series of methodical steps:

## 1. **Finding the Line of Symmetry**
   - **SIFT Feature Matching:**
     - Utilizes SIFT (Scale-Invariant Feature Transform) to extract keypoints and descriptors from both the original image and its horizontally flipped version.
     - Matches these features using a brute-force matcher (`BFMatcher`), selecting the most accurate matches.
   - **Hexbin Analysis:**
     - Analyzes matched points by calculating the angle and distance (`r`, `theta`) relative to the X-axis.
     - Generates a hexbin plot to visualize the distribution of these `(r, θ)` pairs.
   - **Vote Sorting:**
     - Examines the hexbin plot to identify the most prominent bin, indicating the most likely line of symmetry.

## 2. **Finding and Selecting Harris Corners**
   - **Corner Detection:**
     - Applies Harris corner detection to identify points with significant intensity changes, often corresponding to corners.
   - **Thresholding and Sorting:**
     - Filters detected corners using a threshold and selects the top 15 based on their corner response values (corner strength).
   - **Code Implementation:**
     - The `detect_harris_corners` function processes the image, converting it to grayscale, applying the Harris corner detector, and selecting the top 15 corners.

## 3. **Finding Corresponding Points on the Opposite Side**
   - **Symmetry Line Equation:**
     - Determines the line of symmetry (given by `r` and `theta`), and calculates each detected corner's symmetric counterpart using geometric properties of the line.
   - **Opposite Point Calculation:**
     - Computes the corresponding point for each corner on the opposite side of the symmetry line by reflecting the point across the line.

## 4. **Connecting Points Using B-Spline**
   - **Spline Drawing:**
     - Draws a B-spline curve connecting each original corner and its symmetric counterpart.
     - The `draw_b_spline_curve` function interpolates between these points to create smooth connecting curves.

## Final Display
   - The final image displays the original Harris corners, their symmetric counterparts, the symmetry line, and the B-spline curves connecting each pair of points.

---

This approach combines feature matching, symmetry detection, corner detection, and curve interpolation to analyze and visualize symmetry in images.

---

