# coding utf-8

from PIL import Image, ImageDraw, ImageFont
import os

fonts_dir = 'fonts/'
output_dir = 'output/'

characters = open('test.txt', 'r').read().replace('\n', '')


for font_name in list(os.listdir(fonts_dir))[1:]:
    font = ImageFont.truetype(fonts_dir + font_name, 96)

    for output_text in characters:
        img = Image.new(mode='LA', size=(128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), output_text, font=font, fill=(0, 255))

        img.save(output_dir + output_text + '/' +
                 font_name.replace('.ttf', '.png').replace('.ttc', '.png'), "PNG")
