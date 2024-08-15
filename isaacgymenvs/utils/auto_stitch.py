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

def stack_images_vertically_with_labels(directory_path, labels, output_filename, 
                                        space_between=10, font_color="white", 
                                        font_size=40, bold=False):
    """
    Stack all PNG images in the directory vertically in the order of their index,
    and add labels to the top right of each image.

    :param directory_path: Path to the directory containing the images.
    :param labels: List of labels corresponding to each image.
    :param output_filename: Filename for the stacked image.
    :param space_between: Space between images in pixels.
    :param font_color: Color of the label text.
    :param font_size: Size of the label text.
    :param bold: Whether the font should be bold.
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
    font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    font = ImageFont.truetype(font_path, size=font_size)

    # Open all images and determine the total height and max width
    images = [Image.open(os.path.join(directory_path, f)) for f in image_files]
    total_height = sum(img.height for img in images) + space_between * (len(images) - 1)
    max_width = max(img.width for img in images)

    # Create a new blank image with the appropriate size
    stacked_image = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))

    # Paste each image into the stacked image and add labels
    current_y = 0
    for img, label in zip(images, labels):
        stacked_image.paste(img, (0, current_y))
        
        # Add label to the top right of each image
        draw = ImageDraw.Draw(stacked_image)
        text_width, text_height = draw.textsize(label, font=font)
        draw.text((img.width - text_width - 10, current_y + 10), label, font=font, fill=font_color)

        current_y += img.height + space_between

    # Save the stacked image
    stacked_image.save(os.path.join(directory_path, output_filename))

# Example usage
directory_path = './recorded_frames/temp_stitched_frames'
labels = ['2e6 Training Samples', '4e6 Training Samples', '6e6 Training Samples', '8e6 Training Samples', '10e6 Training Samples']  # List of labels corresponding to each image
output_filename = 'AMP_crane_policy_instability.png'

# Modify the settings as needed
stack_images_vertically_with_labels(directory_path, labels, output_filename, 
                                    space_between=15, font_color="white", 
                                    font_size=60, bold=True)
