# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageDraw, ImageFont
import numpy
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.legacy import interfaces
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, PReLU, BatchNormalization, concatenate
from keras.utils import np_utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# define a pooling layer
class GlobalStandardPooling2D(Layer):
    @interfaces.legacy_global_pooling_support
    def __init__(self, data_format=None, **kwargs):
        super(GlobalStandardPooling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.std(inputs, axis=[1, 2])
        else:
            return K.std(inputs, axis=[2, 3])

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(GlobalStandardPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


fonts_dir = './fonts/'
real_img_dir = './raw_img/'
characters = []
raw_images = []
labels = []
fonts_num = 0


for real_img_file in [name for name in os.listdir(real_img_dir) if name != '.DS_Store']:
    characters.append(real_img_file)


for i, font_name in enumerate([name for name in os.listdir(fonts_dir) if name != '.DS_Store']):
    # Read font by using truetype
    font = ImageFont.truetype(fonts_dir + font_name, 96)
    for character in characters:
        # Create a L with alpha img
        img = Image.new(mode='LA', size=(128, 128), color=(255, 0))
        draw = ImageDraw.Draw(img)
        # Make the font drawn on center
        text_size = draw.textsize(character, font)
        text_w = text_size[0]
        text_h = text_size[1]
        draw.text((64 - text_w / 2, 64 - text_h / 2),
                  character, font=font, fill=(0, 255))

        raw_images.append(list(img.getdata()))
        labels.append(i)
        fonts_num = i + 1
print('succeeded: reading fonts')


raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 2).astype('float32') / 255
labels = numpy.array(labels)
labels = np_utils.to_categorical(labels)


# LeNet
input = Input(shape=(128, 128, 2), name='input')
x = Conv2D(input_shape=(128, 128, 2),
           filters=32,
           kernel_size=3,
           padding='same')(input)
x = PReLU()(x)
x = MaxPooling2D(pool_size=2)(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64,
           kernel_size=3,
           padding='same')(x)
x = PReLU()(x)
x = MaxPooling2D(pool_size=2)(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128,
           kernel_size=3,
           padding='same')(x)
x = PReLU()(x)
x = MaxPooling2D(pool_size=2)(x)
x = BatchNormalization()(x)
x = AveragePooling2D(pool_size=2)(x)
x = BatchNormalization()(x)
avgpool = GlobalAveragePooling2D()(x)
stdpool = GlobalStandardPooling2D()(x)
x = concatenate([avgpool, stdpool])
x = Dense(units=256,
          kernel_initializer='random_normal')(x)
x = Dropout(0.25)(x)
x = PReLU()(x)
output = Dense(units=fonts_num,
               kernel_initializer='random_normal',
               activation='softmax',
               name='output')(x)
model = Model(inputs=input, outputs=output)

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(x=raw_images,
                    y=labels,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=32,
                    verbose=2)
