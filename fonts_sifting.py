# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
import os

# Define the abspath
fonts_dir = 'fonts'

# Read 3500 most used chinese characters
characters = open('characters.txt', 'r',
                  encoding='utf-8').read().replace('\n', '')

# Use all fonts in fonts directory
# One item in list is a file named ".DS_Store", not a font file, so ignore it
for font_name in [name for name in os.listdir(fonts_dir) if name[0] != '.']:

    # Read font by using truetype
    font = ImageFont.truetype(fonts_dir + '/' + font_name, 12)

    # Traverse all characters
    for output_text in characters:

        # Create a L image
        img = Image.new(mode='L', size=(16, 16), color=255)

        # Draw the fonts
        draw = ImageDraw.Draw(img)
        draw.text((0,0), output_text, font=font, fill=0)

        # Print the name of fonts which is not applicable
        if img.getextrema()[0] == 255:
            print(font_name)
            break
