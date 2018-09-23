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


# Define model args
l2_rate = 0.01


# Read raw images & characters
raw_images = []
characters = []

for raw_img_file in [name for name in os.listdir(raw_img_dir) if name[0] != '.']:
    for file_name in [name for name in os.listdir(raw_img_dir + os.sep + raw_img_file) if name[0] != '.']:
        raw_images.append(list(Image.open(raw_img_dir + os.sep +
                                          raw_img_file + os.sep + file_name).getdata()))
        characters.append(raw_img_file)


# Process img
raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 2).astype('float32') / 127.5 - 1


# Define the models
generator = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=8,
           kernel_size=3,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=32,
           kernel_size=3,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=64,
           kernel_size=5,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=64,
           kernel_size=5,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=32,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=8,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=2,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same',
           activation='tanh'),
])


# Print model struct
print(generator.summary())

# Load trained models
generator = load_model(model_data_dir + os.sep + 'generator.h5')

# Compile models
generator.compile(loss='logcosh',
                  optimizer='Adadelta',
                  metrics=['acc'])


# Predict image
fake_images = generator.predict(x=raw_images, verbose=1)

for character, fake_image in zip(characters, fake_images):
    save_image = ((fake_image + 1) * 127.5).astype('uint8')
    Image.fromarray(save_image, mode='LA').save(
        fake_img_dir + os.sep + character + '.png')
