# coding utf-8

from PIL import Image, ImageDraw, ImageFont
import os

# Define the abspath
fonts_dir = 'fonts/'
output_dir = 'output/'

# Read 3500 most used chinese characters
characters = open('test.txt', 'r').read().replace('\n', '')

# Use all fonts in fonts' directory
# The first item in list is a file named ".DS_Store", not a text file
for font_name in list(os.listdir(fonts_dir))[1:]:

    # Read font by using truetype
    font = ImageFont.truetype(fonts_dir + font_name, 96)

    # Traverse all characters
    for output_text in characters:

        # Create a L with alpha img
        img = Image.new(mode='LA', size=(128, 128))

        draw = ImageDraw.Draw(img)

        # todo: Make the font draw on center

        draw.text((0, 0), output_text, font=font, fill=(0, 255))

        img.save(output_dir + output_text + '/' +
                 font_name.replace('.ttf', '.png').replace('.ttc', '.png'), "PNG")
