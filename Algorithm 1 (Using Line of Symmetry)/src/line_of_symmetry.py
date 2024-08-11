import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
import pandas as pd

# Create SIFT object
sift = cv2.SIFT_create()
# Create BFMatcher object
bf = cv2.BFMatcher()

class Mirror_Symmetry_detection:
    def __init__(self, image_path: str):
        self.image = self._read_color_image(image_path)
        self.reflected_image = np.fliplr(self.image)
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)
    
    def _read_color_image(self, image_path):
        image = cv2.imread(image_path)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        return image
        
    def find_matchpoints(self):
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        matchpoints = [item[0] for item in matches]
        matchpoints = sorted(matchpoints, key=lambda x: x.distance)
        return matchpoints
    
    def find_points_r_theta(self, matchpoints):
        points_r = []
        points_theta = []
        for match in matchpoints:
            point = self.kp1[match.queryIdx]
            mirpoint = self.kp2[match.trainIdx]
            mirpoint.angle = np.deg2rad(mirpoint.angle)
            mirpoint.angle = np.pi - mirpoint.angle
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2 * np.pi
            mirpoint.pt = (self.reflected_image.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
            theta = angle_with_x_axis(point.pt, mirpoint.pt)
            xc, yc = midpoint(point.pt, mirpoint.pt)
            r = xc * np.cos(theta) + yc * np.sin(theta)
            points_r.append(r)
            points_theta.append(theta)
        return points_r, points_theta

    def draw_matches(self, matchpoints, top=10):
        img = cv2.drawMatches(self.image, self.kp1, self.reflected_image, self.kp2,
                               matchpoints[:top], None, flags=2)
        plt.imshow(img)
        plt.title(f"Top {top} pairs of symmetry points")
        plt.show()
        
    def draw_hex(self, points_r, points_theta):
        image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
        plt.colorbar()
        plt.show()
    
    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
        for k, v in sorted_vote.items():
            if vertical:
                return k[0], k[1]
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]
    
    def sort_hexbin_by_votes(self, image_hexbin):
        counts = image_hexbin.get_array()
        ncnts = np.count_nonzero(np.power(10, counts))
        verts = image_hexbin.get_offsets()
        output = {}
        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc]:
                output[(binx, biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}
                              
    def draw_mirrorLine(self, r, theta, title):
        height, width = self.image.shape[:2]
        x1 = 0
        y1 = int(r / np.sin(theta))
        x2 = width
        y2 = int((r - x2 * np.cos(theta)) / np.sin(theta))
        if y1 < 0:
            y1 = 0
            x1 = int((r - y1 * np.sin(theta)) / np.cos(theta))
        if y2 < 0:
            y2 = 0
            x2 = int((r - y2 * np.sin(theta)) / np.cos(theta))
        cv2.line(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.imshow(self.image)
        plt.axis('off')
        plt.title(title)
        plt.show()
        return x1, y1, x2, y2

def angle_with_x_axis(pi, pj):
    x, y = pi[0] - pj[0], pi[1] - pj[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle

def midpoint(pi, pj):
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2

def read_csv(csv_path):
    data = pd.read_csv(csv_path, header=None)
    curves = data.groupby(0).apply(lambda x: x.iloc[:, 1:].values).tolist()
    return curves

def draw_b_spline_curve(image, points, color=(0, 255, 0)):
    if len(points) < 2:
        return image

    if len(points) == 2:
        cv2.line(image, points[0], points[1], color, 2)
        return image

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    tck, u = splprep([x, y], s=0, k=2)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    for i in range(len(x_new) - 1):
        cv2.line(image, (int(x_new[i]), int(y_new[i])), (int(x_new[i + 1]), int(y_new[i + 1])), color, 2)

    return image

def detecting_mirrorLine(image_path, csv_path, title):
    image = cv2.imread(image_path)
    mirror = Mirror_Symmetry_detection(image_path)
    matchpoints = mirror.find_matchpoints()
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)
    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)
    x1, y1, x2, y2 = mirror.draw_mirrorLine(r, theta, title)

    # Read and plot curves from CSV
    curves = read_csv(csv_path)
    for curve in curves:
        for points in curve:
            image = draw_b_spline_curve(image, points, color=(255, 0, 0))  # Red curves from CSV

    # Detect Harris corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    thresh = 0.001 * dst.max()
    corners = [(i, j) for j in range(dst.shape[0]) for i in range(dst.shape[1]) if dst[j, i] > thresh]
    top_corners = sorted(corners, key=lambda x: dst[x[1], x[0]], reverse=True)[:10]

    for corner in top_corners:
        opposite_point = find_opposite_point(corner, (x1, y1, x2, y2))
        cv2.circle(image, corner, 5, (0, 255, 0), 2)  # Green for the corner
        cv2.circle(image, opposite_point, 5, (0, 0, 255), 2)  # Red for the symmetric point
        image = draw_b_spline_curve(image, [corner, opposite_point], color=(0, 255, 0))

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"{title} - Harris Corners and Symmetric Points")
    plt.show()

def find_opposite_point(corner, line_coords):
    x1, y1, x2, y2 = line_coords
    x, y = corner
    line_vector = np.array([x2 - x1, y2 - y1])
    point_vector = np.array([x - x1, y - y1])
    line_length_squared = np.dot(line_vector, line_vector)
    t = np.dot(point_vector, line_vector) / line_length_squared
    projection = np.array([x1, y1]) + t * line_vector
    opposite_point = 2 * projection - np.array([x, y])
    return tuple(map(int, opposite_point))

if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'
    csv_path = 'path_to_your_curves.csv'
    detecting_mirrorLine(image_path, csv_path, "Mirror Symmetry Detection")
