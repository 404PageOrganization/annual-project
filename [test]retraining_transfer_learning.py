from keras.utils import np_utils
from keras.regularizers import l2
from keras.models import load_model
from keras.layers import Input, BatchNormalization, Conv2D, UpSampling2D, PReLU, Dropout, MaxPooling2D, Activation
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from non_local import non_local_block
from PIL import Image, ImageDraw, ImageFont
import os
import numpy


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define abspaths
fonts_dir = 'raw_fonts'
target_img_dir = 'target_img'
fake_img_dir = 'fake_img'
pretraining_data_dir = 'model_data/pretraining.h5'
model_data_dir = 'model_data/retraining.h5'


# Define hyperparameters
epochs_for_generator = 200
non_trainable_layers = 18
save_image_rate = 10
learning_rate = 0.05
l2_rate = 0.01


# Read target images & characters
target_images = []
characters = []

for target_img_file in [name for name in os.listdir(target_img_dir) if name[0] != '.']:
    for file_name in [name for name in os.listdir('{}/{}'.format(target_img_dir, target_img_file)) if name[0] != '.']:
        target_images.append(list(Image.open('{}/{}/{}'.format(target_img_dir,
                                                               target_img_file, file_name)).getdata()))
        characters.append(target_img_file)


# Make raw images & characters
raw_images = []

# One item in list is a file named ".DS_Store", not a font file, so ignore it
font_list = [name for name in os.listdir(fonts_dir) if name[0] != '.']

# Use 1 font to generate target img
font_name = font_list[0]

# Read font by using truetype
font = ImageFont.truetype('{}/{}'.format(fonts_dir, font_name), 96)

# Traverse all characters
for character in characters:

    # Create a L with alpha img
    img = Image.new(mode='L', size=(128, 128), color=255)

    draw = ImageDraw.Draw(img)

    # Make the font drawn on center
    text_size = draw.textsize(character, font)
    text_w = text_size[0]
    text_h = text_size[1]
    draw.text((64 - text_w / 2, 64 - text_h / 2),
              character, font=font, fill=0)

    raw_images.append(list(img.getdata()))


# Process image
raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1
target_images = numpy.array(target_images)
target_images = target_images.reshape(
    target_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1


# Use only 20 of characters to save
batch_raw_images = raw_images[:20]
batch_target_images = target_images[:20]
batch_characters = characters[:20]


# Save sample images
for character, target_image, raw_image in zip(batch_characters, batch_target_images, batch_raw_images):
    target_sample = ((target_image + 1) *
                     127.5).astype('uint8').reshape(128, 128)
    raw_sample = ((raw_image + 1) * 127.5).astype('uint8').reshape(128, 128)
    Image.fromarray(target_sample, mode='L').save(
        '{}/{}target_img.png'.format(fake_img_dir, character))
    Image.fromarray(raw_sample, mode='L').save(
        '{}/{}raw_img.png'.format(fake_img_dir, character))


# Load pretrained model
generator = load_model(pretraining_data_dir)


# Set some layers to non_trainable
for i, layer in enumerateg(enerator.layers):
    if i > non_trainable_layers:
        break
    else:
        layer.trainable = False


# Print model struct
print(generator.summary())


# Compile models
generator.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['acc'])


# Set callbacks
class auto_save(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Save images
        if((epoch + 1) % save_image_rate == 0):
            print('Saving fake images.')
            fake_images = generator.predict(x=batch_raw_images, verbose=1)
            for character, fake_image in zip(batch_characters, fake_images):
                save_image = ((fake_image + 1) *
                              127.5).astype('uint8').reshape(128, 128)
                Image.fromarray(save_image, mode='L').save(
                    '{}/re{}{}.png'.format(fake_img_dir, character, epoch + 1))

        # Save model
        if (epoch + 1) % save_model_rate == 0:
            print('Saving model...')
            generator.save(model_data_dir)
            print('Saving model succeeded.')


save = auto_save()


# Training generator
generator.fit(x=raw_images,
              y=target_images,
              epochs=epochs_for_generator,
              verbose=2,
              callbacks=[save],
              validation_split=0.2)
