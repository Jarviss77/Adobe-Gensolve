import matplotlib.pyplot as plt
import numpy as np
import cv2

def detect_harris_corners(image_path, thresh_ratio=0.1):
    # Read in the image
    image = cv2.imread(image_path)
    image=np.array(image)

    # Debugging: Check if image is loaded
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Check the file path.")

    # Make a copy of the image
    image_copy = np.array(image)

    # Debugging: Print type and shape of the image
    print(f"Image type: {type(image_copy)}")
    print(f"Image shape: {image_copy.shape}")

    # Change color to RGB (from BGR)
    image_copy = cv2.cvtColor(np.float32(image_copy), cv2.COLOR_BGR2RGB)
    if image_copy.ndim != 3 or image_copy.shape[2] != 3:
        raise ValueError("Image should be a 3-channel (RGB) image.")

    # Convert to grayscale
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)

    # Detect corners
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Dilate corner image to enhance corner points
    dst = cv2.dilate(dst, None)

    # Determine the threshold
    thresh = thresh_ratio * dst.max()

    # Create an image copy to draw corners on
    corner_image = np.copy(image_copy)

    # Find the maximum value in dst and its corresponding coordinates
    for j in range(0, dst.shape[0]):
        for i in range(0, dst.shape[1]):
            if(dst[j,i] > thresh):
            # image, center pt, radius, color, thickness
                cv2.circle( corner_image, (i, j), 1, (0,255,0), 1)

    
    #image_path = '../occlusion2.png'
    #corner_image = detect_harris_corners(image_path)

    plt.imshow(corner_image)
    plt.show()

    return corner_image

detect_harris_corners('../occlusion2.png')

# Usage
#image_path = '../occlusion2.png'
#corner_image = detect_harris_corners(image_path)

#plt.imshow(corner_image)
#plt.show()
