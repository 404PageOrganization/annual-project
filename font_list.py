# coding utf-8

from os import listdir

fonts_dir = 'fonts'
fonts_reserve_dir = 'fonts_reserve'


[name for name in listdir(fonts_reserve_dir) if name != '.DS_Store']

f = open('font_reference.md', 'w')

f.write('### 字体引用目录\n\n')

for font_name in [name for name in listdir(fonts_dir) if name != '.DS_Store']:
    font_name = font_name.replace('.ttf', '').replace('.ttc', '')
    f.write('-   ' + font_name + '\n')

f.write('\n### 备用字体\n\n')

for font_reserve_name in [name for name in listdir(fonts_reserve_dir) if name != '.DS_Store']:
    font_reserve_name = font_reserve_name.replace(
        '.ttf', '').replace('.ttc', '')
    f.write('-   ' + font_reserve_name + '\n')
