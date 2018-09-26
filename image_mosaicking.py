# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageDraw, ImageFont

in_dir = 'fake_img'
out_dir = 'fake_img_mosaicking'
fake_img_name = []
chars = []

img_num = 1
i = 0
rate = 100

for fake_img_file in [name for name in os.listdir(in_dir) if name[0] != '.']:
    fake_img_name.append(fake_img_file)
    name = fake_img_file[0]
    if name not in chars:
        chars.append(name)

row_num = len(fake_img_name) // len(chars)
col_num = 5

img_new = Image.new(mode='L', size=(
    row_num * 128, 64 + col_num * 128), color=255)

for char in chars:
    font = ImageFont.truetype('fonts_reserve/清茶楷体.ttf', 32)
    draw = ImageDraw.Draw(img_new)
    text_w, text_h = draw.textsize('raw_img', font)
    draw.text((64 - text_w / 2, 32 - text_h / 2),
              'raw_img', font=font, fill=0)
    text_w, text_h = draw.textsize('real_img', font)
    draw.text(((row_num - 1) * 128 + 64 - text_w / 2, 32 - text_h / 2),
              'real_img', font=font, fill=0)

    img = Image.open(in_dir + '/' + char + 'raw_img.png')
    img_new.paste(img, (0, 64 + i * 128))
    img = Image.open(in_dir + '/' + char + 'real_img.png')
    img_new.paste(img, ((row_num - 1) * 128, 64 + i * 128))
    for j in range(row_num - 2):
        count = j + 1
        img = Image.open(in_dir + '/' + char + str(count * rate) + '.png')
        img_new.paste(img, (count * 128, 64 + i * 128))
        text_w, text_h = draw.textsize(str(count * rate), font)
        draw.text((count * 128 + 64 - text_w / 2, 32 - text_h / 2),
                  str(count * rate), font=font, fill=0)

    i += 1
    if i == col_num:
        img_new.save(out_dir + '/' + 'mosaicking' +
                     str(img_num) + '.png', 'png')
        img_new = Image.new(mode='L', size=(
            row_num * 128, 64 + col_num * 128), color=255)
        i -= 5
        img_num += 1
