# coding utf-8

import os

output_dir = 'output/'

characters = open('test.txt', 'r').read().replace('\n', '')

for output_text in characters:
    os.system('mkdir ' + output_dir + output_text)
