import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

def speckle_filter(image):
    # Apply a Median filter to reduce speckle noise
    filtered_image = cv2.medianBlur(image, 5)
    return filtered_image

def segment_land_sea(input_image_path, output_image_path):
    # Load the input image
    sar_image = cv2.imread(input_image_path)

    # Apply speckle filtering to the SAR image
    sar_image_filtered = speckle_filter(sar_image)

    # Convert to grayscale if the image has multiple channels
    if len(sar_image_filtered.shape) == 3:
        sar_image_filtered = cv2.cvtColor(sar_image_filtered, cv2.COLOR_BGR2GRAY)

    # Flatten the SAR image to create a 1D array
    data = sar_image_filtered.flatten().reshape(-1, 1)

    # Fit a bimodal Gaussian Mixture Model (GMM) with two components
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(data)

    # Get the estimated means and variances of the two components
    means = gmm.means_.ravel()
    variances = gmm.covariances_.ravel()

    # Determine the threshold between land and sea using the means
    threshold = (means[0] + means[1]) / 2

    # Apply segmentation based on the threshold
    segmented_image = (sar_image_filtered > threshold).astype(np.uint8) * 255

    # Save the segmented image
    cv2.imwrite(output_image_path, segmented_image)

input_base_path = './input/cropped_image'
output_base_path = './results/gmm/speckle_filtered/segmented_image'

for i in range(1, 7):
    input_image_path = f'{input_base_path}{i}.tif'
    output_image_path = f'{output_base_path}{i}.tif'
    segment_land_sea(input_image_path, output_image_path)
