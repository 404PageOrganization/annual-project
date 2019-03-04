# -*- coding: utf-8 -*-

import os
import pickle
from PIL import Image
import numpy
from keras.models import load_model
from custom_layers import GlobalStandardPooling2D


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


target_img_dir = 'target_img/'
target_imgs = []


# Load target images
for target_img_file in [name for name in os.listdir(target_img_dir) if name[0] != '.']:
    img = Image.open(target_img_dir + target_img_file)
    target_imgs.append(list(img.getdata()))

target_imgs = numpy.array(target_imgs)
target_imgs = target_imgs.reshape(
    target_imgs.shape[0], 128, 128, 1).astype('float32') / 255


# Load model and predict the style
model = load_model('model_data/style_discriminator.h5',
                   {'GlobalStandardPooling2D': GlobalStandardPooling2D})

prediction = model.predict(target_imgs)
prediction = numpy.argmax(prediction, axis=1)
prediction = numpy.bincount(prediction)
prediction = numpy.argmax(prediction)
with open('fonts_name.dat', 'rb+') as f:
    fonts_name = pickle.load(f)
print(fonts_name[prediction])
