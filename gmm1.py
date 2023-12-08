import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

def speckle_filter(image):
    if image is not None:
        # Apply a Median filter to reduce speckle noise
        filtered_image = cv2.medianBlur(image, 5)
        return filtered_image
    else:
        print("Error: Input image is None")
        return None

def region_growing_denoise(segmented_image, neighborhood_size):
    height, width = segmented_image.shape
    denoised_image = np.zeros_like(segmented_image)

    for i in range(height):
        for j in range(width):
            if segmented_image[i, j] == 255:
                white_count = 0
                black_count = 0

                # Count white and black pixels in the local neighborhood
                for x in range(i - neighborhood_size, i + neighborhood_size + 1):
                    for y in range(j - neighborhood_size, j + neighborhood_size + 1):
                        if 0 <= x < height and 0 <= y < width:
                            if segmented_image[x, y] == 255:
                                white_count += 1
                            else:
                                black_count += 1

                # Adjust the pixel color based on the density of white and black pixels
                if white_count > black_count:
                    denoised_image[i, j] = 255
                else:
                    denoised_image[i, j] = 0

    return denoised_image

def segment_land_sea(input_image_path, output_image_path, neighborhood_size):
    sar_image = cv2.imread(input_image_path)

    if sar_image is not None:
        sar_image_filtered = speckle_filter(sar_image)

        if sar_image_filtered is not None:
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

            # Denoise the segmented image using region growing
            denoised_image = region_growing_denoise(segmented_image, neighborhood_size)

            # Save the denoised segmented image
            cv2.imwrite(output_image_path, denoised_image)

            print(f"Processed {input_image_path} and saved to {output_image_path}")
        else:
            print("Denoising failed. Check the speckle_filter function.")
    else:
        print(f"Failed to load the image: {input_image_path}")

input_base_path = './input/cropped_image'
output_base_path = './results/gmm/speckle_filtered_region_growing/corrected_image'
neighborhood_size = 3  # Adjust this value based on your experimentation

# for i in range(1, 7):
#     input_image_path = f'{input_base_path}{i}.tif'
#     output_image_path = f'{output_base_path}{i}.tif'
#     segment_land_sea(input_image_path, output_image_path, neighborhood_size)
for i in range(1,21):
    segment_land_sea(input_base_path+'1.tif', f'{output_base_path}{i}.tif', i)