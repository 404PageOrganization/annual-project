# coding utf-8

from PIL import Image, ImageDraw, ImageFont
import os

# Define the abspath
fonts_dir = 'fonts'
raw_img_dir = 'raw_img'

# Read 3500 most used chinese characters
characters = open('test.txt', 'r', encoding='utf-8').read().replace('\n', '')

# Use all fonts in fonts' directory
# One item in list is a file named ".DS_Store", not a font file, so ignore it
for font_name in [name for name in os.listdir(fonts_dir) if name != '.DS_Store']:

    # Read font by using truetype
    font = ImageFont.truetype(fonts_dir + os.sep + font_name, 96)

    # Traverse all characters
    for output_text in characters:

        # Create a L with alpha img
        img = Image.new(mode='LA', size=(128, 128))

        draw = ImageDraw.Draw(img)

        # Make the font drawn on center
        text_size = draw.textsize(output_text, font)
        text_w = text_size[0]
        text_h = text_size[1]
        draw.text((64 - text_w / 2, 64 - text_h / 2),
                  output_text, font=font, fill=(0, 255))

        # Save the image
        img.save(raw_img_dir + os.sep + output_text + os.sep +
                 font_name.replace('.ttf', '.png').replace('.ttc', '.png'), "PNG")
