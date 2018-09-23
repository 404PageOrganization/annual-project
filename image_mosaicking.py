# -*- coding: utf-8 -*-

import os
from PIL import Image

in_dir = './fake_img/'
out_dir = './fake_img_mosaicking/'
fake_img_name = []
chars = []

img_num = 1
i = 0

for fake_img_file in [name for name in os.listdir(in_dir) if name != '.DS_Store']:
    fake_img_name.append(fake_img_file)
    name = fake_img_file[0]
    if name not in chars:
        chars.append(name)

row_num = len(fake_img_name) // len(chars)
col_num = 5

img_new = Image.new(mode='L', size=(row_num * 128, col_num * 128), color=255)

for char in chars:
    img = Image.open(in_dir + char + 'raw_img.png')
    img_new.paste(img, (0, i * 128))
    img = Image.open(in_dir + char + 'real_img.png')
    img_new.paste(img, ((row_num - 1) * 128, i * 128))
    for j in range(row_num - 2):
        img = Image.open(in_dir + char + str((j + 1) * 10) + '.png')
        img_new.paste(img, ((j + 1) * 128, i * 128))
    i += 1
    if i == col_num:
        img_new.save(out_dir + 'mosaicking' + str(img_num) + '.png', 'png')
        img_new = Image.new(mode='L', size=(row_num * 128, col_num * 128), color=255)
        i -= 5
        img_num += 1