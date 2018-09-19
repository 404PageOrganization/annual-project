from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, LeakyReLU, ELU
from PIL import Image
import os
import numpy

raw_img_dir = 'raw_img'

raw_images = []
real_images = []
fake_images = []
images = []
label = []

# Read raw images
for i, raw_img_file in enumerate([name for name in os.listdir(raw_img_dir) if name != '.DS_Store']):
    for file_name in [name for name in os.listdir(raw_img_dir + os.sep + raw_img_file) if name != '.DS_Store']:
        real_images.append(list(Image.open(raw_img_dir + os.sep +
                                           raw_img_file + os.sep + file_name).getdata()))


'''
x = numpy.array(images)
y = numpy.array(label)

x = x.reshape(x.shape[0], 128, 128, 2).astype('float32') / 255
y = np_utils.to_categorical(y)
'''

Generative = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=12,
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
    Dense(input_shape=(49152,),
          units=2048,
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
    Reshape((32, 32, 48), input_shape=(49152,)),
    UpSampling2D(2),
    LeakyReLU(alpha=0.3),
    Conv2D(input_shape=(32, 32, 48),
           filters=24,
           kernel_size=3,
           padding='same'),
    UpSampling2D(2),
    LeakyReLU(alpha=0.3),
    Conv2D(filters=12,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Conv2D(filters=2,
           kernel_size=3,
           padding='same',
           activation='softmax'),
])

Discriminative = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=12,
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

print(Generative.summary())
print(Discriminative.summary())
