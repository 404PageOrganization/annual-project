import os

base_dir = '/Users/adelard/Documents/annual_project/'
output_dir = '/Users/adelard/Documents/annual_project/output/'

characters = open(base_dir + 'characters.txt', 'r').read().replace('\n', '')

for output_text in characters:
    os.system('mkdir ' + output_dir + output_text)
