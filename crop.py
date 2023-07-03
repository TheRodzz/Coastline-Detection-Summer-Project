# this code is used to crop SAR image in a tif file
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def crop_tiff_image(input_path, output_path, left, upper, right, lower):
    # Open the TIFF image
    img = Image.open(input_path)

    # Crop the image using the coordinates (left, upper, right, lower)
    cropped_img = img.crop((left, upper, right, lower))

    # Save the cropped image to the output path
    cropped_img.save(output_path)

if __name__ == "__main__":
    # Specify the input TIFF file path
    input_tiff_file = "ICEYE_GRD_SC_744762_20220624T210044.tif"

    # Specify the output TIFF file path
    output_tiff_file = "cropped_image.tif"

    # Specify the coordinates for cropping (left, upper, right, lower)
    crop_left = 14608# The x-coordinate of the left edge of the crop box
    crop_upper = 18448# The y-coordinate of the upper edge of the crop box
    crop_right = 17744# The x-coordinate of the right edge of the crop box
    crop_lower = 22416# The y-coordinate of the lower edge of the crop box

    # Crop the image
    crop_tiff_image(input_tiff_file, output_tiff_file, crop_left, crop_upper, crop_right, crop_lower)

