import cv2
import matplotlib.pyplot as plt
import numpy as np

# Create a SIFT detector object
sift = cv2.SIFT_create()

# Hardcoded values
image_path = 'butterfly.png'
output_image_path = 'butterfly-symmetry.png'

def very_close(a, b, tol=4.0):
    """Checks if the points a, b are within tol distance of each other."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < tol

def S(si, sj, sigma=1):
    """Computes the 'S' function mentioned in the research paper."""
    q = (-abs(si - sj)) / (sigma * (si + sj))
    return np.exp(q ** 2)

def reisfeld(phi, phj, theta):
    return 1 - np.cos(phi + phj - 2 * theta)

def midpoint(i, j):
    return (i[0] + j[0]) / 2, (i[1] + j[1]) / 2

def angle_with_x_axis(i, j):
    x, y = i[0] - j[0], i[1] - j[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan2(y, x)
    return angle

def superm2(image):
    """Performs the symmetry detection on image."""
    mimage = np.fliplr(image)
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(mimage, None)
    for p, mp in zip(kp1, kp2):
        p.angle = np.deg2rad(p.angle)
        mp.angle = np.deg2rad(mp.angle)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    houghr = []
    houghth = []
    weights = []
    for match, match2 in matches:
        point = kp1[match.queryIdx]
        mirpoint = kp2[match.trainIdx]
        mirpoint2 = kp2[match2.trainIdx]
        mirpoint2.angle = np.pi - mirpoint2.angle
        mirpoint.angle = np.pi - mirpoint.angle
        if mirpoint.angle < 0.0:
            mirpoint.angle += 2 * np.pi
        if mirpoint2.angle < 0.0:
            mirpoint2.angle += 2 * np.pi
        mirpoint.pt = (mimage.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
        if very_close(point.pt, mirpoint.pt):
            mirpoint = mirpoint2
        theta = angle_with_x_axis(point.pt, mirpoint.pt)
        xc, yc = midpoint(point.pt, mirpoint.pt)
        r = xc * np.cos(theta) + yc * np.sin(theta)
        Mij = reisfeld(point.angle, mirpoint.angle, theta) * S(
            point.size, mirpoint.size
        )
        houghr.append(r)
        houghth.append(theta)
        weights.append(Mij)

    houghr = np.array(houghr)
    houghth = np.array(houghth)
    weights = np.array(weights)

    # Find the most prominent line of symmetry
    max_weight_index = np.argmax(weights)
    best_r = houghr[max_weight_index]
    best_theta = houghth[max_weight_index]

    def draw(image, r, theta):
        """Draws a line of symmetry on the image based on r and theta."""
        if np.pi / 4 < theta < 3 * (np.pi / 4):
            for x in range(image.shape[1]):
                y = int((r - x * np.cos(theta)) / np.sin(theta))
                if 0 <= y < image.shape[0]:
                    image[y, x] = 255
        else:
            for y in range(image.shape[0]):
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < image.shape[1]:
                    image[y, x] = 255

    def hex():
        plt.hexbin(houghr, houghth, C=weights, bins='log')
        plt.colorbar()
        plt.show()

    hex()
    draw(image, best_r, best_theta)
    cv2.imshow('Symmetry Detection', image)
    cv2.waitKey(0)
    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to load image from {image_path}")
    else:
        superm2(image)
