# coding utf-8

from PIL import Image, ImageDraw, ImageFont
import os

base_dir = '/Users/adelard/Documents/annual_project/'
fonts_dir = '/Users/adelard/Documents/annual_project/fonts/'
output_dir = '/Users/adelard/Documents/annual_project/output/'

characters = open(base_dir + 'test.txt', 'r').read().replace('\n', '')


for font_name in list(os.listdir(fonts_dir))[1:]:
    font = ImageFont.truetype(fonts_dir + font_name, 96)

    for output_text in characters:
        img = Image.new(mode='L', size=(128, 128), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), output_text, font=font)

        img.save(output_dir + output_text + '/' +
                 font_name.replace('.ttf', '') + '.png', "PNG")
