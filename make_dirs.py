# -*- coding: utf-8 -*-

import os

dirs = ['fake_img', 'fake_img_mosaicking', 'fonts_reserve', 'model_data', 'output_img', 'raw_fonts', 'target_fonts', 'target_img', 'target_img_origin']

for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)
