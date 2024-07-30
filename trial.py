# import numpy as np
# def read_csv ( csv_path ):
#     np_path_XYs = np . genfromtxt ( csv_path , delimiter = ',')
#     path_XYs = []
#     for i in np . unique ( np_path_XYs [: , 0]):
#         npXYs = np_path_XYs [ np_path_XYs [: , 0] == i ][: , 1:]
#         XYs = []
#         for j in np . unique ( npXYs [: , 0]):
#             XY = npXYs [ npXYs [: , 0] == j ][: , 1:]
#             XYs . append ( XY )
#         path_XYs . append ( XYs )
#     return path_XYs

# import numpy as np
# import matplotlib . pyplot as plt

# def plot ( paths_XYs ):
#     fig , ax = plt . subplots ( tight_layout = True , figsize =(8 , 8))
#     for i , XYs in enumerate ( paths_XYs ):
#         # c = colours [ i % len( colours )]
#         for XY in XYs :
#             ax . plot ( XY [: , 0] , XY [: , 1] , linewidth =2)
#     ax . set_aspect ( "equal")
#     plt . show ()

# a = read_csv("occlusion1.csv")
# # plot(a)

# import numpy as np
# import svgwrite
# import cairosvg

# def polylines2svg(paths_XYs, svg_path):
#     W, H = 0, 0
#     for path_XYs in paths_XYs:
#         for XY in path_XYs:
#             W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
#     padding = 0.1
#     W, H = int(W + padding * W), int(H + padding * H)
    
#     # Create a new SVG drawing
#     dwg = svgwrite.Drawing(svg_path, profile="tiny", shape_rendering="crispEdges")
#     group = dwg.g()
#     colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]

#     for i, path in enumerate(paths_XYs):
#         path_data = []
#         for XY in path:
#             path_data.append("M {} {}".format(XY[0, 0], XY[0, 1]))
#             for j in range(1, len(XY)):
#                 path_data.append("L {} {}".format(XY[j, 0], XY[j, 1]))
#             if not np.allclose(XY[0], XY[-1]):
#                 path_data.append("Z")
#         c = colors[i%len(colors)]        
#         group.add(dwg.path(d=" ".join(path_data), fill=c, stroke="none", stroke_width=2))

#     dwg.add(group)
#     dwg.save()
    
#     png_path = svg_path.replace('.svg', '.png')
#     fact = 1
#     if min(H,W)!=0: fact = max(1, 1024 // min(H, W))
    
#     # cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')
# # print(a[:1])
# polylines2svg(a, "hello.svg")import pandas as pd

import pandas as pd
import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline, interp1d

# Smoothing function
def smooth_points(x, y, s=0):
    spline_x = UnivariateSpline(range(len(x)), x, s=s)
    spline_y = UnivariateSpline(range(len(y)), y, s=s)
    return spline_x(range(len(x))), spline_y(range(len(y)))

# Interpolation function
def interpolate_points(x, y, num_points):
    t = np.linspace(0, 1, len(x))
    f_x = interp1d(t, x, kind='linear')
    f_y = interp1d(t, y, kind='linear')
    t_new = np.linspace(0, 1, num_points)
    return f_x(t_new), f_y(t_new)

# Convert points to image
def points_to_image(points, width=1000, height=1000):
    img = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        if 0 <= int(y) < height and 0 <= int(x) < width:
            img[int(y), int(x)] = 255
    return img

# Detect shapes
def detect_shapes(img):
    shapes = []
    edges = cv2.Canny(img.copy(), 0, 50)
    edges_line = cv2.GaussianBlur(edges.copy(), (15, 15), 0)
    

    # Detect lines using Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges_line, 0.01, np.pi/2 , threshold=200, minLineLength=0, maxLineGap=100)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                shapes.append(("Line", np.array([[x1, y1], [x2, y2]])))
                
    edges = img.copy()
    # Find contours
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 500:  # Adjust the threshold as needed
            continue

        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            shapes.append(("Triangle", approx))
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape = "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"
            shapes.append((shape, approx))
        elif len(approx) > 4:
            area = cv2.contourArea(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = area / (np.pi * radius * radius)
            if 0.7 <= circularity <= 1.3:
                center = (int(x), int(y))
                radius = int(radius)
                shapes.append(("Circle", (center, radius)))
            else:
                shapes.append(("Polygon", approx))

            # Check for ellipse
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                shapes.append(("Ellipse", cv2.ellipse2Poly(
                    center=(int(ellipse[0][0]), int(ellipse[0][1])),
                    axes=(int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    angle=int(ellipse[2]),
                    arcStart=0,
                    arcEnd=360,
                    delta=5
                )))

            # Check for star
            if len(approx) >= 10:
                shapes.append(("Star", approx))

    # Select the shape with the highest probability
    shape_priorities = {"Circle": 1, "Square": 2, "Rectangle": 3, "Triangle": 4, "Star": 5, "Polygon": 6, "Ellipse": 7, "Line": 8}
    if shapes:
        shapes = sorted(shapes, key=lambda s: shape_priorities.get(s[0], 9))
        most_probable_shape = shapes[0]
        return [most_probable_shape]
    return shapes

def draw_shapes(img, shapes, curve_points=None):
    # Convert the image to color if it is grayscale
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()
    
    # Create a blank image with the same dimensions as the input image
    blank_image = np.zeros_like(img_color)
    
    if shapes:
        for shape, contour in shapes:
            color = (255, 255, 255)  # White
            
            if shape == "Circle":
                center, radius = contour
                cv2.circle(blank_image, center, radius, color, 1)
            else:
                # Draw the shape on the blank image
                cv2.drawContours(blank_image, [contour], -1, color, 1)
    else:
        # Draw the original curve if no shapes are detected
        if curve_points is not None:
            color = (255, 255, 255)  # White
            cv2.polylines(blank_image, [curve_points], isClosed=False, color=color, thickness=1)
    
    return blank_image


# Function to combine images into a single large image
def combine_images(images, positions, width=1000, height=1000):
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    for img, (x, y) in zip(images, positions):
        h, w = img.shape[:2]
        x = max(0, min(x, width - w))  # Ensure x is within bounds
        y = max(0, min(y, height - h))  # Ensure y is within bounds
        mask = img != 0
        combined_image[y:y+h, x:x+w][mask] = img[mask]
    return combined_image

# Process each curve
processed_curves = []
images = []
positions = []

# Load data into a DataFrame
df = pd.read_csv("occlusion1.csv", header=None, names=['Curve', 'Shape', 'X', 'Y'])


# Group by curve
curves = df.groupby('Curve')
for curve_id, group in curves:
    x, y = group['X'].values, group['Y'].values
    x_smooth, y_smooth = smooth_points(x, y, s=0)
    x_interp, y_interp = interpolate_points(x_smooth, y_smooth, num_points=1000)

    points = np.vstack((x_interp, y_interp)).T
    positions.append((int(x.min()), int(y.min())))  # Store original positions for combining

    img = points_to_image(points)
    # cv2.imwrite(f"without_shapes_detected_{curve_id}.png", img)
    shapes = detect_shapes(img)

    # If no shapes are detected, use the original curve points
    img_with_shapes = draw_shapes(img, shapes, curve_points=np.int32(points))
    images.append(img_with_shapes)
    # cv2.imwrite(f"dummy_shapes_detected_{curve_id}.png", img_with_shapes)

# Combine all images into one large image
combined_image = combine_images(images, positions, width=1000, height=1000)
cv2.imwrite("combined_shapes.png", combined_image)