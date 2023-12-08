import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


def speckle_filter(image):
    # Apply a Median filter to reduce speckle noise
    filtered_image = cv2.medianBlur(image, 5)
    return filtered_image


def calculate_density(segmented_image, i, j, neighborhood_size):
    height, width = segmented_image.shape
    white_count = 0

    for x in range(i - neighborhood_size, i + neighborhood_size + 1):
        for y in range(j - neighborhood_size, j + neighborhood_size + 1):
            if 0 <= x < height and 0 <= y < width:
                if segmented_image[x, y] == 255:
                    white_count += 1

    # Calculate and return the density
    return white_count / (neighborhood_size * neighborhood_size)


def segment_land_sea(sar_image):
    # Load the input image
    # sar_image = cv2.imread(input_image_path)

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

    # Calculate weighted threshold based on means and variances
    weights = gmm.weights_
    threshold = (weights[0] * (means[0] + np.sqrt(variances[0])) + weights[1] * (means[1] + np.sqrt(variances[1]))) / (
                weights[0] + weights[1])

    # Apply segmentation based on the threshold
    segmented_image = (sar_image_filtered > threshold).astype(np.uint8) * 255

    # Use a neighborhood size of 3x3 to calculate the density of white pixels and color those pixels
    neighborhood_size = 3
    height, width = sar_image_filtered.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if segmented_image[i, j] == 255:
                # Calculate the density of white pixels in the 3x3 neighborhood
                density = calculate_density(segmented_image, i, j, neighborhood_size)

                # Color the pixel based on density
                if density > 0.5:
                    segmented_image[i, j] = 255  # Set pixel to white
                else:
                    segmented_image[i, j] = 0  # Set pixel to black

    # Save the segmented image
    # cv2.imwrite(output_image_path, segmented_image
    return segmented_image


input_base_path = './input/cropped_image'
output_base_path = './results/multiple-iterations/segmented_image'

for i in range(1, 7):
    input_image_path = f'{input_base_path}{i}.tif'
    output_image_path = f'{output_base_path}{i}.tif'

    sar_image = cv2.imread(input_image_path)
    for _ in range(4):
        sar_image = segment_land_sea(sar_image)
    cv2.imwrite(output_image_path, sar_image)