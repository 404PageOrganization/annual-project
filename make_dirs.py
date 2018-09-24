# -*- coding: utf-8 -*-

import os

dirs = ['fonts', 'fonts_reserve', 'raw_img', 'real_img_origin',
        'real_img', 'fake_img', 'fake_img_mosaicking', 'model_data']

for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)
