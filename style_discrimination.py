# -*- coding: utf-8 -*-

import os
import pickle
from PIL import Image, ImageDraw, ImageFont
import random
import numpy
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, PReLU, BatchNormalization, concatenate
from custom_layers import GlobalStandardPooling2D
from keras.utils import np_utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


fonts_dir = 'fonts/'
raw_img_dir = 'raw_img/'
characters = []
fonts_name = []

BATCH = 256


# read characters
characters = open('characters.txt', 'r',
                  encoding='utf-8').read()
characters = list(characters)
random.shuffle(characters)
char_num = len(characters)
epoch_num = char_num // BATCH + 1
print('succeeded: reading characters')
print(epoch_num)


# read fonts name
for i, font_name in enumerate([name for name in os.listdir(fonts_dir) if name[0] != '.']):
    # output number of fonts
    fonts_num = i + 1
    print(str(i), font_name.replace('.ttf', ''))
    fonts_name.append(font_name)
with open('fonts_name.dat', 'rb+') as f:
    pickle.dump(fonts_name, f)
print('succeeded: reading fonts name')


# the model
input = Input(shape=(128, 128, 1), name='input')
x = Conv2D(input_shape=(128, 128, 1),
           filters=16,
           kernel_size=3,
           padding='same')(input)
x = PReLU()(x)
x = MaxPooling2D(pool_size=2)(x)
x = BatchNormalization()(x)
x = Conv2D(filters=32,
           kernel_size=3,
           padding='same')(x)
x = PReLU()(x)
x = MaxPooling2D(pool_size=2)(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64,
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
x = Dense(units=128,
          kernel_initializer='random_normal')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
output = Dense(units=fonts_num,
               kernel_initializer='random_normal',
               activation='softmax',
               name='output')(x)
model = Model(inputs=input, outputs=output)

# print model
print(model.summary())

# compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


for epoch in range(epoch_num):
    if os.path.exists('model_data/style_discriminator.h5'):
        model = load_model('model_data/style_discriminator.h5')
    #
    raw_imgs = []
    labels = []
    if epoch == epoch_num:
        characters_using = characters
    else:
        characters_using = characters[:BATCH - 1]
        characters = characters[BATCH:]

    #
    for i, font_name in enumerate(fonts_name):
        # read font by using truetype
        font = ImageFont.truetype(fonts_dir + font_name, 96)
        for character in characters_using:
            # create an img
            img = Image.new(mode='L', size=(128, 128), color=255)
            draw = ImageDraw.Draw(img)
            # make the font drawn on center
            text_size = draw.textsize(character, font)
            text_w = text_size[0]
            text_h = text_size[1]
            draw.text((64 - text_w / 2, 64 - text_h / 2),
                      character, font=font, fill=0)

            raw_imgs.append(list(img.getdata()))
            labels.append(i)

    # randomize dataset
    dataset = list(zip(raw_imgs, labels))
    random.shuffle(dataset)
    raw_imgs[:], labels[:] = zip(*dataset)
    print('succeeded: randomizing data set')

    # tranfer the dataset into a numpy array
    raw_imgs = numpy.array(raw_imgs)
    raw_imgs = raw_imgs.reshape(
        raw_imgs.shape[0], 128, 128, 1).astype('float32') / 255
    labels = numpy.array(labels)
    labels = np_utils.to_categorical(labels)

    # train model
    history = model.fit(x=raw_imgs,
                        y=labels,
                        validation_split=0.2,
                        initial_epoch=epoch * 100,
                        epochs=100,
                        batch_size=128,
                        verbose=2)

    # save model
    model.save('model_data/style_discriminator.h5')
