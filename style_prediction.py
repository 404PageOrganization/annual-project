# -*- coding: utf-8 -*-

import os
from PIL import Image
import numpy
from keras.models import load_model
from custom_layers import GlobalStandardPooling2D


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


real_img_dir = 'real_img/'
real_imgs = []

for real_img_file in [name for name in os.listdir(real_img_dir) if name[0] != '.']:
    img = Image.open(real_img_dir + real_img_file)
    real_imgs.append(list(img.getdata()))

real_imgs = numpy.array(real_imgs)
real_imgs = real_imgs.reshape(
    real_imgs.shape[0], 128, 128, 2).astype('float32') / 255

model = load_model('model_data/style_discriminator.h5',
                   {'GlobalStandardPooling2D': GlobalStandardPooling2D})

prediction = model.predict(real_imgs)
prediction = numpy.argmax(prediction, axis=1)  # axis = 1是取行
print(prediction)
