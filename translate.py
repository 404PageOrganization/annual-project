from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, LeakyReLU, ELU
from PIL import Image
import os
import numpy

# Define abspaths
raw_img_dir = 'raw_img'
real_img_dir = 'real_img'
fake_img_dir = 'fake_img'

epochs = 1
epochs_combine = 1

raw_images = []
real_images = []
characters = []

# Read raw images
for raw_img_file in [name for name in os.listdir(raw_img_dir) if name != '.DS_Store']:
    for file_name in [name for name in os.listdir(raw_img_dir + os.sep + raw_img_file) if name != '.DS_Store']:
        raw_images.append(list(Image.open(raw_img_dir + os.sep +
                                          raw_img_file + os.sep + file_name).getdata()))
        characters.append(raw_img_file)

# Read real images
for real_img_file in [name for name in os.listdir(real_img_dir) if name != '.DS_Store']:
    for file_name in [name for name in os.listdir(real_img_dir + os.sep + real_img_file) if name != '.DS_Store']:
        real_images.append(list(Image.open(real_img_dir + os.sep +
                                           real_img_file + os.sep + file_name).getdata()))

raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 1).astype('float32') / 255
real_images = numpy.array(real_images)
real_images = real_images.reshape(
    real_images.shape[0], 128, 128, 1).astype('float32') / 255

generator = Sequential([
    Conv2D(input_shape=(128, 128, 1),
           filters=6,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=24,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=48,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=49152,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=2048,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=128,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=128,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=2048,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=49152,
          kernel_initializer='normal',
          activation='relu'),
    Reshape((32, 32, 48),
            input_shape=(49152,)),
    UpSampling2D(2),
    LeakyReLU(alpha=0.3),
    Conv2D(input_shape=(32, 32, 48),
           filters=24,
           kernel_size=3,
           padding='same'),
    UpSampling2D(2),
    LeakyReLU(alpha=0.3),
    Conv2D(filters=6,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Conv2D(filters=1,
           kernel_size=3,
           padding='same',
           activation='softmax'),
])

discriminator = Sequential([
    Conv2D(input_shape=(128, 128, 1),
           filters=6,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=24,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=48,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=2048,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dropout(0.5),
    Dense(units=16,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=2,
          kernel_initializer='normal',
          activation='softmax')
])

# Connect generator with discriminator
discriminator.trainable = False
combine = Sequential([generator, discriminator])

# Print model struct
print(generator.summary())
print(discriminator.summary())
print(combine.summary())

# Compile models
generator.compile(loss='categorical_crossentropy', optimizer='adam')
discriminator.compile(loss='categorical_crossentropy', optimizer='adam')
combine.compile(loss='categorical_crossentropy', optimizer='adam')

# Train models
for epoch in range(1, epochs + 1):
    print('Epoch:{}'.format(epoch))

    fake_images = generator.predict(raw_images)

    for real, fake in zip(real_images, fake_images):

        images = numpy.array((real, fake))

        y = [1, 0]
        y = numpy.array(y)
        y = np_utils.to_categorical(y)
        discriminator.trainable = True
        discriminator.fit(images, y, verbose=0)

    discriminator.trainable = False

    length = len(characters)
    y = [[0, 1]] * length
    y = numpy.array(y).astype('float32')

    for i in range(epochs_combine):
        combine.fit(raw_images, y, verbose=0)
