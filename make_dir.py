# coding utf-8

import os
import platform

# Define the abspath
output_dir = 'output/'

# Read 3500 most used chinese characters
characters = open('test.txt', 'r').read().replace('\n', '')

# Make all directorys
for output_text in characters:
    if 'Windows' in platform.platform():
        os.system('md' + output_dir + output_text)
    else:
        os.system('mkdir ' + output_dir + output_text)
