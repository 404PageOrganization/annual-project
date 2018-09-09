# coding utf-8

import os
import platform

# Define the abspath
output_dir = 'output'

# Read 3500 most used chinese characters
characters = open('test.txt', 'r', encoding='utf-8').read().replace('\n', '')

# Make all directorys
for output_text in characters:
    os.mkdir(output_dir + os.sep + output_text)
