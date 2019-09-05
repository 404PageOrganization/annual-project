# coding utf-8

from PIL import Image, ImageDraw, ImageFont
import os

# Define the abspath
fonts_dir = 'target_fonts'
img_dir = 'target_img'

# Use font to generate target image
characters = open('characters.txt', 'r',
                  encoding='utf-8').read().replace('\n', '')

# Use all fonts in fonts' directory
# One item in list is a file named '.DS_Store', not a font file, so ignore it
for font_name in [name for name in os.listdir(fonts_dir) if name[0] != '.']:

    # Read font by using truetype
    font = ImageFont.truetype('{}/{}'.format(fonts_dir, font_name), 96)

    # Traverse all characters
    for output_text in characters:

        # Create a grayscale img
        img = Image.new(mode='L', size=(128, 128), color=255)

        draw = ImageDraw.Draw(img)

        # Make the font drawn on center
        text_size = draw.textsize(output_text, font)
        text_w = text_size[0]
        text_h = text_size[1]
        draw.text((64 - text_w / 2, 64 - text_h / 2),
                  output_text, font=font, fill=0)

        # Save the image
        img.save('{}/{}.png'.format(img_dir, output_text), 'PNG')