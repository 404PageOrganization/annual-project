from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
import os
import numpy


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define abspaths
fonts_dir = 'raw_fonts'
fake_img_dir = 'predict_img'
model_data_dir = 'model_data/generator.h5'


# Read all characters
characters = open('characters.txt', 'r',
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
raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1


# Load trained models
generator = load_model(model_data_dir)


# Print model struct
print(generator.summary())


# Predict image
fake_images = generator.predict(x=raw_images, verbose=1)

for character, fake_image in zip(characters, fake_images):
    save_image = ((fake_image + 1) * 127.5).astype('uint8').reshape(128, 128)
    Image.fromarray(save_image, mode='L').save(
        fake_img_dir + '/' + character + '.png')
