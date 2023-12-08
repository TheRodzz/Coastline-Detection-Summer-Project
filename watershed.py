import cv2
import numpy as np
from skimage import color
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

def speckle_filter(image):
    # Apply a Median filter to reduce speckle noise
    filtered_image = cv2.medianBlur(image, 5)
    return filtered_image

def segment_land_sea(input_image_path, output_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)

    # Apply speckle filtering
    filtered_image = speckle_filter(image)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Perform thresholding to obtain a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create markers for the Watershed algorithm
    markers = np.zeros_like(gray, dtype=np.int32)
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)

    # Apply the Watershed algorithm
    cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # Mark the boundary as red

    # Save the segmented image
    cv2.imwrite(output_image_path, image)

# Define input and output base paths
input_base_path = './input/cropped_image'
output_base_path = './results/watershed/speckle_filtered/segmented_image'

# Process multiple images
for i in range(1, 7):
    input_image_path = f'{input_base_path}{i}.tif'
    output_image_path = f'{output_base_path}{i}.tif'
    segment_land_sea(input_image_path, output_image_path)
