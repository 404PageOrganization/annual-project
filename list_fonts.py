# coding utf-8

import os
import glob

# Define directories here
fonts_dir = 'fonts'
fonts_reserve_dir = 'fonts_reserve'

# Open the markdown file
f = open('font_reference.md', 'w', encoding='utf-8')

f.write('### 字体引用目录\n\n')

# Write font reference
for font_name in glob.glob(fonts_dir + '/*.ttf'):
    font_name = font_name.replace(
        fonts_dir + '/', '').replace('.ttf', '')
    f.write('-   ' + font_name + '\n')

f.write('\n### 备用字体\n\n')

# Write font_reserve reference
for font_reserve_name in glob.glob(fonts_reserve_dir + '/*.ttf'):
    font_reserve_name = font_reserve_name.replace(
        fonts_reserve_dir + '/', '').replace('.ttf', '')
    f.write('-   ' + font_reserve_name + '\n')
