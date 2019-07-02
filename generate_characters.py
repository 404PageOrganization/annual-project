from keras.models import load_model, Model
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define abspaths
fonts_dir = 'raw_fonts'
generated_img_dir = 'generated_img'
model_data_dir = 'model_data/gan_model.h5'


# Define Normal Scale
norm_scale = 0.14


# Read generate characters
characters = open('test.txt', 'r',
                  encoding='utf-8').read().replace('\n', '')


# Make raw images & characters
raw_images = []

# One item in list is a file named ".DS_Store", not a font file, so ignore it
font_list = [name for name in os.listdir(fonts_dir) if name[0] != '.']

# Use 1 font to generate target img
font_name = font_list[0]

# Read font by using truetype
font = ImageFont.truetype('{}/{}'.format(fonts_dir, font_name), 96)

# Traverse all characters
for character in characters:

    # Create a grayscale img
    img = Image.new(mode='L', size=(128, 128), color=255)

    draw = ImageDraw.Draw(img)

    # Make the font drawn on center
    text_size = draw.textsize(character, font)
    text_w = text_size[0]
    text_h = text_size[1]
    draw.text((64 - text_w / 2, 64 - text_h / 2),
              character, font=font, fill=0)

    raw_images.append(list(img.getdata()))


# Process image
raw_images = np.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1
raw_images += np.random.normal(
    size=(raw_images.shape[0], 128, 128, 1), scale=norm_scale)


# Load trained models
gan = load_model(model_data_dir)
generator = Model(inputs=gan.input, outputs=gan.layers[1].get_output_at(-1))


# Print model struct
print(generator.summary())


# Predict image
generated_images = generator.predict(x=raw_images, verbose=1)

for character, generated_image in zip(characters, generated_images):
    save_image = ((generated_image + 1) *
                  127.5).astype('uint8').reshape(128, 128)
    Image.fromarray(save_image, mode='L').save(
        '{}/{}.png'.format(generated_img_dir, character))
