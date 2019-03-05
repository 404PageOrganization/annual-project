# -*- coding: utf-8 -*-

import os

dirs = ['raw_fonts', 'target_fonts', 'fonts_reserve', 'target_img_origin',
        'target_img', 'fake_img', 'fake_img_mosaicking', 'model_data']

for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)
