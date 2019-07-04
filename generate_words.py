from PIL import Image, ImageFont, ImageDraw
from keras.models import load_model, Model
from random import uniform
import colorama
import numpy as np
import os
import sys


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


colorama.init(autoreset=True)


# Define abspaths
fonts_dir = 'raw_fonts'
output_img_dir = 'output_img'
words_dir = 'test.txt'
model_data_dir = 'model_data/gan_model.h5'


# Define default Args
args = {
    'characters_per_line': 20,
    'lines_per_page': 20,
    'transform_div': 0.07,
    'norm_scale': 0.14
}


# Read Argv
argv = sys.argv[1:]
if argv[0] in ('-h', '--help'):
    print('Available args: %s' % ', '.join(args.keys()))
    exit()

for k, v in zip(argv[0::2], argv[1::2]):
    assert k[:2] == '--', colorama.Fore.RED + \
        colorama.Style.BRIGHT + 'Argument %s is not start with --!' % k
    assert k[2:] in args, colorama.Fore.RED + \
        colorama.Style.BRIGHT + 'Arg name %s not find!' % k[2:]

    args[k[2:]] = eval(v)


# One item in list is a file named '.DS_Store', not a font file, so ignore it
font_list = [name for name in os.listdir(fonts_dir) if name[0] != '.']

# Use 1 font to generate target img
font_name = font_list[0]

# Read font by using truetype
font = ImageFont.truetype('{}/{}'.format(fonts_dir, font_name), 96)


# Load trained models
gan = load_model(model_data_dir)
generator = Model(inputs=gan.input, outputs=gan.layers[1].get_output_at(-1))


print(colorama.Fore.GREEN + colorama.Style.BRIGHT + 'Model loaded successfully.')


# Read generate words
sentences = open(words_dir, 'r',
                 encoding='utf-8').read().split('\n')


# Make output
page = 0
lines = 0
y_pos = 0

result = Image.new(mode='L', size=(
    128 * args['characters_per_line'], 128 * args['lines_per_page']), color=255)


for sentence in sentences:
    # Init variables
    x_pos = 0
    raw_images = []

    # Traverse all characters
    for character in sentence:

        # Create a grayscale img
        img = Image.new(mode='L', size=(128, 128), color=255)

        draw = ImageDraw.Draw(img)

        # Make the font drawn on center
        text_size = draw.textsize(character, font)
        text_w = text_size[0]
        text_h = text_size[1]
        draw.text((64 - text_w / 2, 64 - text_h / 2),
                  character, font=font, fill=0)

        data = [uniform(1-args['transform_div'], 1+args['transform_div']), uniform(-args['transform_div'], args['transform_div']), uniform(-args['transform_div'], args['transform_div']),
                uniform(-args['transform_div'], args['transform_div']), uniform(1-args['transform_div'], 1+args['transform_div']), uniform(-args['transform_div'], args['transform_div'])]

        img_trans = img.transform(
            (128, 128), Image.AFFINE, data, fillcolor=255)

        raw_images.append(list(img_trans.getdata()))

    # Process image
    raw_images = np.array(raw_images)
    raw_images = raw_images.reshape(
        raw_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1
    raw_images += np.random.normal(
        size=(raw_images.shape[0], 128, 128, 1), scale=args['norm_scale'])

    # Predict image
    generated_images = generator.predict(x=raw_images, verbose=0)

    # Typesetting
    for generated_image in generated_images:
        generated_image = ((generated_image + 1) *
                           127.5).astype('uint8').reshape(128, 128)
        generated_image = Image.fromarray(generated_image).convert('L')

        if x_pos == args['characters_per_line']:
            x_pos = 0
            y_pos += 1

        if y_pos == args['lines_per_page']:
            # Save page and reset y pos
            result.save('{}/output{}.png'.format(output_img_dir, page), 'PNG')
            y_pos = 0
            page += 1
            # Create a new page
            result = Image.new(mode='L', size=(
                128 * args['characters_per_line'], 128 * args['lines_per_page']), color=255)

        result.paste(generated_image, (128 * x_pos, 128 * y_pos))

        x_pos += 1

    y_pos += 1

# Save final page
result.save('{}/output{}.png'.format(output_img_dir, page), 'PNG')
print(colorama.Fore.GREEN + colorama.Style.BRIGHT + 'Output successfully.')
