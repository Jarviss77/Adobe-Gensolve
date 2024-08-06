import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from untitled2 import Mirror_Symmetry_detection
import random

def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    shapes = []

    for shape_id in np.unique(data[:, 0]):
        shape_points = data[data[:, 0] == shape_id][:, 2:]  # Skip the polyline_id and point_id columns
        shapes.append(shape_points)
    
    return shapes

def detect_harris_corners(image, thresh_ratio=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    thresh = thresh_ratio * dst.max()

    corner_image = np.copy(image)
    points_with_values = []
    for j in range(dst.shape[0]):
        for i in range(dst.shape[1]):
            if dst[j, i] > thresh:
                points_with_values.append((dst[j, i], (i, j)))

    points_with_values.sort(reverse=True, key=lambda x: x[0])
    top_points = [point[1] for point in points_with_values[:20]]
    
    for i, j in top_points:
        cv2.circle(corner_image, (i, j), 1, (0, 255, 0), 1)

    return top_points, corner_image

def draw_spline_curve(image, points, color):
    if len(points) < 2:
        return image

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    tck, _ = splprep([x, y], s=0, k=min(len(points)-1, 3))
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)

    for i in range(len(out[0]) - 1):
        cv2.line(image, (int(out[0][i]), int(out[1][i])), (int(out[0][i + 1]), int(out[1][i + 1])), color, 1)
    
    return image

def detecting_mirrorLine(csv_path, title, show_detail=False):
    shapes = read_csv(csv_path)
    
    for shape_index, shape in enumerate(shapes):
        print(f"Processing shape {shape_index + 1}/{len(shapes)}")

        # Convert shape points to an image
        image = np.zeros((500, 500, 3), dtype=np.uint8)  # Create a blank image
        for point in shape:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 1, (255, 255, 255), 1)

        corners, corner_image = detect_harris_corners(image)

        mirror = Mirror_Symmetry_detection(image)
        matchpoints = mirror.find_matchpoints()
        points_r, points_theta = mirror.find_points_r_theta(matchpoints)

        if show_detail:
            mirror.draw_matches(matchpoints, top=10)
            mirror.draw_hex(points_r, points_theta)

        image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r) 
        sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
        r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)  

        symmetry_line = mirror.draw_mirrorLine(r, theta, title)

        if symmetry_line:
            x1, y1, x2, y2 = symmetry_line
            symmetry_pairs = []
            tolerance = 1  

            distance_dict = {}
            for corner in corners:
                x, y = corner
                distance = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                matched = False
                for d in distance_dict:
                    if np.abs(distance - d) < tolerance:
                        distance_dict[d].append(corner)
                        matched = True
                        break
                if not matched:
                    distance_dict[distance] = [corner]

            for distance, pts in distance_dict.items():
                if len(pts) >= 2:
                    for i in range(len(pts)):
                        for j in range(i + 1, len(pts)):
                            p1, p2 = pts[i], pts[j]
                            dist_between_points = np.linalg.norm(np.array(p1) - np.array(p2))
                            if dist_between_points > distance:
                                symmetry_pairs.append((p1, p2))

            for p1, p2 in symmetry_pairs:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                corner_image = draw_spline_curve(corner_image, [p1, p2], color)

        plt.imshow(corner_image)
        plt.axis('off')
        plt.title(f"Shape {shape_index + 1}")
        plt.show()

    return

def main():
    detecting_mirrorLine('.csv', "butterfly with Mirror Line", show_detail=True)

if __name__ == '__main__':
    main()
