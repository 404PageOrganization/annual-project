from keras.regularizers import l2
from keras.models import Model
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
target_img_dir = 'pretraining_target_img'
fake_img_dir = 'pretraining_fake_img'
model_data_dir = 'model_data/pretraining.h5'


# Define hyperparameters
epochs_for_generator = 200
save_model_rate = 50
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


# One item in list is a file named ".DS_Store", not a font file, so ignore it
font_list = [name for name in os.listdir(fonts_dir) if name[0] != '.']

# Use 1 font to generate target img
font_name = font_list[0]

# Read font by using truetype
font = ImageFont.truetype('{}/{}'.format(fonts_dir, font_name), 96)

# Use only 20 of characters to save
batch_raw_images = []
batch_characters = characters[:20]

# Traverse batch characters
for character in batch_characters:

    # Create a grayscale img
    img = Image.new(mode='L', size=(128, 128), color=255)

    draw = ImageDraw.Draw(img)

    # Make the font drawn on center
    text_size = draw.textsize(character, font)
    text_w = text_size[0]
    text_h = text_size[1]
    draw.text((64 - text_w / 2, 64 - text_h / 2),
              character, font=font, fill=0)

    batch_raw_images.append(list(img.getdata()))

# Process image
batch_raw_images = numpy.array(batch_raw_images)
batch_raw_images = batch_raw_images.reshape(
    batch_raw_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1
target_images = numpy.array(target_images)
target_images = target_images.reshape(
    target_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1

batch_target_images = target_images[:20]

# Save sample images
for character, target_image, raw_image in zip(batch_characters, batch_target_images, batch_raw_images):
    target_sample = ((target_image + 1) *
                     127.5).astype('uint8').reshape(128, 128)
    raw_sample = ((raw_image + 1) * 127.5).astype('uint8').reshape(128, 128)
    Image.fromarray(target_sample, mode='L').save(
        '{}/{}target_img.png'.format(fake_img_dir, character))
    Image.fromarray(raw_sample, mode='L').save(
        '{}/{}raw_img.png'.format(fake_img_dir, character))


# Define the models
input = Input(shape=(128, 128, 1), name='input')
x = Conv2D(input_shape=(128, 128, 1),
           filters=8,
           kernel_size=64,
           padding='same')(input)
#x = non_local_block(x, compression=2, mode='embedded')
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=8,
           kernel_size=64,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=32,
           kernel_size=32,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=32,
           kernel_size=32,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64,
           kernel_size=16,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64,
           kernel_size=16,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128,
           kernel_size=7,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128,
           kernel_size=3,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128,
           kernel_size=3,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128,
           kernel_size=3,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(filters=1,
           kernel_size=3,
           padding='same')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(filters=1,
           kernel_size=3,
           padding='same')(x)
output = Activation('tanh', name='output')(x)
generator = Model(inputs=input, outputs=output)


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
                    '{}/{}pre{}.png'.format(fake_img_dir, character, epoch + 1))

        # Save model
        if (epoch + 1) % save_model_rate == 0:
            print('Saving model...')
            generator.save(model_data_dir)
            print('Saving model succeeded.')


save = auto_save()


# Dynamic generate training data
def generate_training_data(font, characters, target_images, batch_size):
    while 1:
        ans = 0
        X = []
        Y = []
        characters
        for character, target_image in zip(characters, target_images):
            img = Image.new(mode='L', size=(128, 128), color=255)
            draw = ImageDraw.Draw(img)
            # Make the font drawn on center
            text_size = draw.textsize(character, font)
            text_w = text_size[0]
            text_h = text_size[1]
            draw.text((64 - text_w / 2, 64 - text_h / 2),
                      character, font=font, fill=0)

            X.append(list(img.getdata()))
            Y.append(target_image)
            ans += 1

            if ans == batch_size:
                ans = 0
                X = numpy.array(X)
                X = X.reshape(X.shape[0], 128, 128, 1).astype(
                    'float32') / 127.5 - 1
                yield (X, numpy.array(Y))
                X = []
                Y = []


# Training generator
generator.fit_generator(generate_training_data(font, characters, target_images, 32),
                        steps_per_epoch=len(characters) // 32,
                        epochs=epochs_for_generator,
                        verbose=2,
                        callbacks=[save])
