import os
from PIL import Image

def crop_center(image_path, crop_width, crop_height):
    """
    Crop the center of the image.

    :param image_path: Path to the image to crop.
    :param crop_width: Width of the crop box.
    :param crop_height: Height of the crop box.
    :return: Cropped image.
    """
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate the coordinates of the crop box
    left = 90 + (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

def crop_images_in_directory(directory_path, crop_width, crop_height):
    """
    Crop all PNG images in a directory and save them with '_crop' appended to the filename.

    :param directory_path: Path to the directory containing the images.
    :param crop_width: Width of the crop box.
    :param crop_height: Height of the crop box.
    """
    # Ensure the directory path ends with a slash
    if not directory_path.endswith('/'):
        directory_path += '/'

    # List all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.png'):
            # Full path to the image file
            image_path = os.path.join(directory_path, filename)
            
            # Crop the image
            cropped_img = crop_center(image_path, crop_width, crop_height)
            
            # Save the cropped image with '_crop' appended to the filename
            base_name, ext = os.path.splitext(filename)
            new_filename = f"{base_name}_crop{ext}"
            cropped_img.save(os.path.join(directory_path, new_filename))

def stitch_images(directory_path, output_filename):
    """
    Stitch all cropped images in the directory in the order of their frame numbers.

    :param directory_path: Path to the directory containing the cropped images.
    :param output_filename: Filename for the stitched image.
    """
    # Ensure the directory path ends with a slash
    if not directory_path.endswith('/'):
        directory_path += '/'

    # List all cropped image files in the directory
    cropped_files = [f for f in os.listdir(directory_path) if f.lower().endswith('_crop.png')]

    # Sort the files based on the frame number
    cropped_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

    # Open all images and determine the total width and max height
    images = [Image.open(os.path.join(directory_path, f)) for f in cropped_files]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the appropriate size
    stitched_image = Image.new('RGB', (total_width, max_height))

    # Paste each image into the stitched image
    current_x = 0
    for img in images:
        stitched_image.paste(img, (current_x, 0))
        current_x += img.width

    # Save the stitched image
    stitched_image.save(os.path.join(directory_path, output_filename))

# Example usage
directory_path = './crane_10M/selected_frames'
crop_width = 300  # Width of the crop box
crop_height = 500  # Height of the crop box
output_filename = 'stitched_image.png'

crop_images_in_directory(directory_path, crop_width, crop_height)
stitch_images(directory_path, output_filename)
