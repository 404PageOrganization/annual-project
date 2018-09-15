# coding utf-8

import os

# Define directories here
fonts_dir = 'fonts'
fonts_reserve_dir = 'fonts_reserve'

# Open the markdown file
f = open('font_reference.md', 'w')

f.write('### 字体引用目录\n\n')

# Write font reference
for font_name in [name for name in os.listdir(fonts_dir) if name != '.DS_Store']:
    font_name = font_name.replace('.ttf', '').replace('.ttc', '')
    f.write('-   ' + font_name + '\n')

f.write('\n### 备用字体\n\n')

# Write font_reserve reference
for font_reserve_name in [name for name in os.listdir(fonts_reserve_dir) if name != '.DS_Store']:
    font_reserve_name = font_reserve_name.replace(
        '.ttf', '').replace('.ttc', '')
    f.write('-   ' + font_reserve_name + '\n')
