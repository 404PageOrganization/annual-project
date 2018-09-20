# coding utf-8

import os

# Define the abspath
raw_img_dir = 'raw_img'

# Read 3500 most used chinese characters
characters = open('test.txt', 'r', encoding='utf-8').read().replace('\n', '')

# Make all directories
for output_text in characters:
    dir = raw_img_dir + os.sep + output_text
    if not os.path.exists(dir):
        os.mkdir(dir)
