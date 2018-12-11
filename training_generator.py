from keras.utils import np_utils
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, PReLU
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from PIL import Image, ImageDraw, ImageFont
import os
import numpy


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define abspaths
fonts_dir = 'fonts'
raw_img_dir = 'raw_img'
real_img_dir = 'real_img'
fake_img_dir = 'fake_img'
model_data_dir = 'model_data/model.h5'


# Define hyperparameters
epochs_for_generator = 200
save_image_rate = 1
learning_rate = 0.05
l2_rate = 0.01


# Read real images & characters
real_images = []
characters = []

for real_img_file in [name for name in os.listdir(real_img_dir) if name[0] != '.']:
    for file_name in [name for name in os.listdir(real_img_dir + '/' + real_img_file) if name[0] != '.']:
        real_images.append(list(Image.open(real_img_dir + '/' +
                                           real_img_file + '/' + file_name).getdata()))
        characters.append(real_img_file)


# Make raw images & characters
raw_images = []

# One item in list is a file named ".DS_Store", not a font file, so ignore it
font_list = [name for name in os.listdir(fonts_dir) if name[0] != '.']

# Use 1 font to generate real img
font_name = font_list[0]

# Read font by using truetype
font = ImageFont.truetype(fonts_dir + '/' + font_name, 96)

# Traverse all characters
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


# Process image
raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 2).astype('float32') / 127.5 - 1
real_images = numpy.array(real_images)
real_images = real_images.reshape(
    real_images.shape[0], 128, 128, 2).astype('float32') / 127.5 - 1


# Use only 20 of characters to save
batch_raw_images = raw_images[:20]
batch_real_images = real_images[:20]
batch_characters = characters[:20]

# Save sample images
for character, real_image, raw_image in zip(batch_characters, batch_real_images, batch_raw_images):
    real_sample = ((real_image + 1) * 127.5).astype('uint8')
    raw_sample = ((raw_image + 1) * 127.5).astype('uint8')
    Image.fromarray(real_sample, mode='LA').save(
        fake_img_dir + '/' + character + 'real_img.png')
    Image.fromarray(raw_sample, mode='LA').save(
        fake_img_dir + '/' + character + 'raw_img.png')


# Define the models
generator = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=8,
           kernel_size=3,
           strides=2,
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=16,
           kernel_size=3,
           strides=2,
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=32,
           kernel_size=5,
           strides=2,
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=32,
           kernel_size=5,
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=16,
           kernel_size=3,
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=8,
           kernel_size=3,
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=2,
           kernel_size=3,
           kernel_regularizer=l2(l2_rate),
           padding='same',
           activation='tanh'),
])


# Print model struct
print(generator.summary())


# Compile models
generator.compile(loss='logcosh',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['acc'])


# Set callbacks
checkpoint = ModelCheckpoint(model_data_dir, save_best_only=True)


class save_fake_img(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(epoch % save_image_rate == 0):
            print('Saving fake images.')
            fake_images = generator.predict(x=batch_raw_images, verbose=1)
            for character, fake_image in zip(batch_characters, fake_images):
                save_image = ((fake_image + 1) * 127.5).astype('uint8')
                Image.fromarray(save_image, mode='LA').save(
                    fake_img_dir + '/' + character + str(epoch) + '.png')


save_img = save_fake_img()


# Training generator
generator.fit(x=raw_images,
              y=real_images,
              epochs=epochs_for_generator,
              verbose=2,
              callbacks=[checkpoint, save_img],
              validation_split=0.2)
