import os
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

def extract_index(filename):
    """
    Extract the index from the filename using regex.
    
    :param filename: The filename to extract the index from.
    :return: The extracted index as an integer.
    """
    match = re.search(r'stitched_image_(\d+)M', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")

def stack_images_vertically_with_labels(directory_path, labels, output_filename):
    """
    Stack all PNG images in the directory vertically in the order of their index,
    and add labels to the top right of each image.

    :param directory_path: Path to the directory containing the images.
    :param labels: List of labels corresponding to each image.
    :param output_filename: Filename for the stacked image.
    """
    # Ensure the directory path ends with a slash
    if not directory_path.endswith('/'):
        directory_path += '/'

    # List all relevant image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().startswith('stitched_image') and f.lower().endswith('.png')]

    # Sort the files based on the index
    image_files.sort(key=extract_index)

    # Check that the number of labels matches the number of images
    if len(labels) != len(image_files):
        raise ValueError("The number of labels does not match the number of images.")

    # Load the standard font from matplotlib
    font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
    font = ImageFont.truetype(font_path, size=40)

    # Open all images and determine the total height and max width
    images = [Image.open(os.path.join(directory_path, f)) for f in image_files]
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    # Create a new blank image with the appropriate size
    stacked_image = Image.new('RGB', (max_width, total_height))

    # Paste each image into the stacked image and add labels
    current_y = 0
    for img, label in zip(images, labels):
        stacked_image.paste(img, (0, current_y))
        
        # Add label to the top right of each image
        draw = ImageDraw.Draw(stacked_image)
        text_width, text_height = draw.textsize(label, font=font)
        draw.text((img.width - text_width - 10, current_y + 10), label, font=font, fill="white")

        current_y += img.height

    # Save the stacked image
    stacked_image.save(os.path.join(directory_path, output_filename))

# Example usage
directory_path = './crane_final_stitched_images'
labels = ['2e6 Samples', '4e6 Samples', '6e6 Samples', '8e6 Samples', '10e6 Samples']  # List of labels corresponding to each image
output_filename = 'stacked_image_with_labels.png'

stack_images_vertically_with_labels(directory_path, labels, output_filename)
