from keras.utils import np_utils
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, PReLU
from PIL import Image
import os
import numpy


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define abspaths
fonts_dir = 'fonts'
raw_img_dir = 'raw_img'
fake_img_dir = 'fake_img'
model_data_dir = 'model_data'


# Read raw images & characters
raw_images = []
characters = []

for raw_img_file in [name for name in os.listdir(raw_img_dir) if name[0] != '.']:
    for file_name in [name for name in os.listdir(raw_img_dir + '/' + raw_img_file) if name[0] != '.']:
        raw_images.append(list(Image.open(raw_img_dir + '/' +
                                          raw_img_file + '/' + file_name).getdata()))
        characters.append(raw_img_file)


# Process img
raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 2).astype('float32') / 127.5 - 1


# Load trained models
generator = load_model(model_data_dir + '/generator.h5')


# Print model struct
print(generator.summary())


# Predict image
fake_images = generator.predict(x=raw_images, verbose=1)

for character, fake_image in zip(characters, fake_images):
    save_image = ((fake_image + 1) * 127.5).astype('uint8')
    Image.fromarray(save_image, mode='L').save(
        fake_img_dir + '/' + character + '.png')
