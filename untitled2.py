import cv2
import numpy as np
import matplotlib.pyplot as plt

def fit_circle(points):
    """ Fit a circle to the given points. """
    if len(points) < 3:
        return None

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_centered = x - x_mean
    y_centered = y - y_mean

    A = np.vstack([x_centered**2 + y_centered**2, x_centered, y_centered, np.ones_like(x_centered)]).T
    B = np.zeros(A.shape[0])

    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    A1, B1, C1, D1 = C
    xc = -A1 / 2
    yc = -B1 / 2
    r = np.sqrt((A1 + B1**2 / 4 - C1) / 4 + (D1 - xc**2 - yc**2) / 4)

    return xc + x_mean, yc + y_mean, r

def fit_ellipse(points):
    """ Fit an ellipse to the given points. """
    if len(points) < 5:
        return None

    points = np.array(points, dtype=np.float32)
    ellipse = cv2.fitEllipse(points)
    
    return ellipse

def fit_convex_hull(points):
    """ Fit a convex hull to the given points. """
    if len(points) < 3:
        print("Error: At least 3 points are required to compute a convex hull.")
        return None

    points = np.array(points, dtype=np.float32)

    if len(points) < 3:
        print("Error: Less than 3 points are provided.")
        return None

    # Ensure points are in the correct shape for cv2.convexHull
    points = points.reshape(-1, 1, 2)

    try:
        hull = cv2.convexHull(points)
        hull = hull.reshape(-1, 2)
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return None

    return hull

def draw_shape(image, points, shape_type):
    """ Draw fitted shapes on the image. """
    if shape_type == 'circle':
        result = fit_circle(points)
        if result:
            xc, yc, r = result
            cv2.circle(image, (int(xc), int(yc)), int(r), (0, 255, 0), 2)
    elif shape_type == 'ellipse':
        result = fit_ellipse(points)
        if result:
            center, axes, angle = result
            cv2.ellipse(image, center, (int(axes[0]), int(axes[1])), angle, 0, 360, (0, 255, 0), 2)
    elif shape_type == 'convex_hull':
        result = fit_convex_hull(points)
        if result is not None:
            # Ensure result is in the correct format for polylines
            result = result.reshape(-1, 1, 2)
            cv2.polylines(image, [result], isClosed=True, color=(0, 255, 0), thickness=2)

def main():
    # Sample data points
    points = [(50, 50), (100, 50), (75, 100), (60, 80)]
    
    # Create a blank image
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Draw shapes
    draw_shape(image, points, 'convex_hull')
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
