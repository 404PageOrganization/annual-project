from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from PIL import Image
import os
import numpy

output_dir = 'output'

font_len = 13
characters_count = 40

images = []
label = []

for i, output_file in enumerate([name for name in os.listdir(output_dir) if name != '.DS_Store']):
    for file_name in [name for name in os.listdir(output_dir + os.sep + output_file) if name != '.DS_Store']:
        images.append(list(Image.open(output_dir + os.sep +
                                      output_file + os.sep + file_name).getdata()))
        label.append(i)

x = numpy.array(images)
y = numpy.array(label)

x = x.reshape(x.shape[0], 128, 128, 2).astype('float32') / 255
y = np_utils.to_categorical(y)

Generative = Sequential([

])

Discriminative = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=12,
           kernel_size=3,
           padding='same',
           activation='relu'),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=24,
           kernel_size=3,
           padding='same',
           activation='relu'),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=48,
           kernel_size=3,
           padding='same',
           activation='relu'),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=1600,
          kernel_initializer='normal',
          activation='relu'),
    Dropout(0.5),
    Dense(units=400,
          kernel_initializer='normal',
          activation='relu'),
    Dense(units=40,
          kernel_initializer='normal',
          activation='softmax')
])
