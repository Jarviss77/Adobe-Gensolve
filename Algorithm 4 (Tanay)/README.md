# Shape Detection and Regularization and Completion (Algorithm 4)

## Sample output
![frag1](..\outputs\algo4\frag1.png)
![isolated](outputs\algo4\isolated.png)
![occlusion1](outputs\algo4\occlusion1.png)
![occlusion2](outputs\algo4\occlusion2.png)


## Overview

This algorithm processes a series of polylines from a CSV file, detecting the closest possible shapes using OpenCV's contour analysis techniques. It is designed to identify shapes such as circles, ellipses, straight lines, polygons, and stars within the given curves. If no shape is detected, the original shape (curve) is retained. The identified shapes or curves are then visualized on an image, and finally, all the images are combined into one composite image.


## Algorithm Workflow

1. **Data Loading:** The algorithm begins by loading the CSV data containing polyline coordinates.
  
2. **Smoothing and Interpolation:** Each polyline is smoothed and interpolated to refine the curve, providing a higher resolution for shape detection.

3. **Shape Detection:** The smoothed and interpolated data is converted into an image, where OpenCV is used to detect predefined shapes. If no shape is recognized, the original curve is retained.

4. **Shape Drawing:** Detected shapes or original curves are drawn on a blank image, and their coordinates are recorded.

5. **Image Combination:** All individual images are combined into a final composite image, maintaining their relative positions, which is then saved as an output file.

## Shape Detection

The shape detection process in this algorithm involves several steps to accurately identify and classify geometric shapes from a set of polylines. Here’s a breakdown of how shape detection is performed:

1. **Data Preparation:**
   - **Smoothing:** Each polyline is smoothed to reduce noise and create a more continuous curve. This is done using `UnivariateSpline`, which fits a spline to the data points. The smoothing parameter `s` controls the degree of smoothness.
   - **Interpolation:** The smoothed curves are then interpolated to increase the resolution of the data. This interpolation creates a high-resolution set of points along the curve, which helps in better shape detection.

2. **Image Conversion:**
   - The interpolated points are converted into a binary image where each point is represented as a white pixel on a black background. This conversion helps in using OpenCV’s shape detection functions.

3. **Edge Detection and Shape Detection:**
   - **Edge Detection:** The Canny edge detector is applied to find edges in the image, which helps in identifying the boundaries of shapes. A Gaussian blur is applied to smooth the edges and reduce noise.
   - **Line Detection:** Using the Hough Line Transform, straight lines in the image are detected. Detected lines are stored as coordinates.
   - **Contour Detection:** Contours of shapes are found using `cv2.findContours`. Contours represent the boundaries of shapes detected in the image.
   
4. **Shape Approximation:**
   - **Polygon Approximation:** For each detected contour, `cv2.approxPolyDP` is used to approximate the contour to a polygon with fewer vertices. This helps in simplifying the shape and making it easier to classify.
   - **Shape Classification:**
     - **Circle:** A shape is classified as a circle if it has a high degree of circularity. Circularity is calculated by comparing the area of the shape to the area of the circle that would have the same radius.
     - **Ellipse:** If the contour is close to an ellipse, it is classified as an ellipse. The `cv2.fitEllipse` function helps in fitting an ellipse to the contour.
     - **Triangle, Square, Rectangle:** The number of vertices in the approximated polygon helps in classifying the shape. For example, three vertices indicate a triangle, four vertices suggest a square or rectangle, and so on.
     - **Star:** A shape with exactly ten vertices is classified as a star.
     - **Polygon:** Shapes with more than four vertices but not fitting the criteria for other shapes are classified as polygons.

5. **Shape Drawing:**
   - Detected shapes are drawn on a blank image using OpenCV drawing functions. Shapes such as circles are approximated by generating points around the perimeter, while polygons and contours are drawn directly.

6. **Shape Prioritization:**
   - Shapes are sorted based on a predefined priority list. This ensures that the most likely shape is selected and displayed. The priority order is: Circle, Square, Rectangle, Triangle, Ellipse, Star, Polygon, and Line.

By following these steps, the algorithm accurately detects and classifies geometric shapes, providing a visual representation of the identified shapes or retaining the original curves when no recognizable shape is found.

This shape detection process is robust and versatile, making it suitable for a wide range of applications where precise shape recognition is crucial.
