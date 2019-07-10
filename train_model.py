from PIL import Image, ImageDraw, ImageFont
from keras.callbacks import ModelCheckpoint, Callback
from keras.initializers import RandomNormal
from keras.layers import Input, Reshape, Dropout, Concatenate, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from lib.image_mosaicking import image_mosaick
import colorama
import datetime
import numpy as np
import os


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


colorama.init(autoreset=True)


# Define abspaths
fonts_dir = 'raw_fonts'
target_img_dir = 'target_img'
fake_img_dir = 'fake_img'
predict_img_dir = 'predict_img'
model_data_dir = 'model_data/gan_model.h5'


# Define hyperparameters
epochs_for_gan = 150
batch_size = 32
save_model_rate = 50
save_image_rate = 10
learning_rate = 0.0001
l2_rate = 0.01
df = 64
gf = 64
norm_scale = 0.07

# Read target images & characters
target_images = []
characters = []

for target_img_file in os.listdir(target_img_dir):
    if target_img_file[0] == '.':
        continue
    target_images.append(
        list(Image.open('{}/{}'.format(target_img_dir, target_img_file)).getdata()))
    characters.append(target_img_file.split('.')[0])


# One item in list is a file named '.DS_Store', not a font file, so ignore it
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
batch_raw_images = np.array(batch_raw_images)
batch_raw_images = batch_raw_images.reshape(
    batch_raw_images.shape[0], 128, 128, 1).astype('float32') / 127.5 - 1
target_images = np.array(target_images)
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


# Define U-Net generator
def build_generator():
    def conv2d(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same',
                   kernel_initializer=RandomNormal(
                       mean=0.0, stddev=0.05, seed=None),
                   kernel_regularizer=l2(l2_rate))(layer_input)
        d = PReLU()(d)
        if bn:
            d = BatchNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                   padding='same', activation='relu',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Input raw image
    d0 = Input(shape=(128, 128, 1))

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf * 8)
    u2 = deconv2d(u1, d5, gf * 8)
    u3 = deconv2d(u2, d4, gf * 8)
    u4 = deconv2d(u3, d3, gf * 4)
    u5 = deconv2d(u4, d2, gf * 2)
    u6 = deconv2d(u5, d1, gf)
    u7 = UpSampling2D(size=2)(u6)

    # Output fake image
    output_img = Conv2D(1, kernel_size=4,
                        strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)


# Define PatchGAN discriminator
def build_discriminator():
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(layer_input)
        d = PReLU()(d)
        if bn:
            d = BatchNormalization()(d)
        return d

    # img_A is target image or fake image, and img_B is raw_img
    img_A = Input(shape=(128, 128, 1))
    img_B = Input(shape=(128, 128, 1))

    # Concatenate two images by channel to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    # Output
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)


# Complie GAN model
discriminator = build_discriminator()
discriminator.compile(loss='mse',
                           optimizer=Adam(lr=learning_rate),
                           metrics=['acc'])
discriminator.trainable = False
generator = build_generator()

img_A = Input(shape=(128, 128, 1))
img_B = Input(shape=(128, 128, 1))
fake_A = generator(img_B)
validity = discriminator([fake_A, img_B])

gan = Model(inputs=[img_A, img_B], outputs=[validity, fake_A])
gan.compile(loss=['mse', 'mae'],
            loss_weights=[1, 100],
            optimizer=Adam(lr=learning_rate))

print(colorama.Fore.GREEN + colorama.Style.BRIGHT +
      'Model compiled successfully.')


# Dynamically generate training data
def generate_training_data(font, characters, target_images, batch_size):
    it = 0
    ans = 0
    X = []
    Y = []

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
            X = np.array(X)
            X = X.reshape(X.shape[0], 128, 128, 1).astype(
                'float32') / 127.5 - 1
            X += np.random.normal(size=(X.shape[0],
                                        128, 128, 1), scale=norm_scale)

            yield (X, np.array(Y))
            X = []
            Y = []
            it += 1
            if it == len(characters) // batch_size:
                return


# Save sample images
def save_image(epoch):
    print('Saving fake images...')
    fake_images = generator.predict(x=batch_raw_images, verbose=1)
    for character, fake_image in zip(batch_characters, fake_images):
        save_image = ((fake_image + 1) *
                      127.5).astype('uint8').reshape(128, 128)
        Image.fromarray(save_image, mode='L').save(
            '{}/{}{}.png'.format(fake_img_dir, character, epoch + 1))
    print(colorama.Fore.GREEN + colorama.Style.BRIGHT +
          'Images saved successfully.')


# Save model
def save_model():
    print('Saving model...')
    gan.save(model_data_dir)
    print(colorama.Fore.GREEN + colorama.Style.BRIGHT +
          'Model saved successfully.')


# Train GAN
def train():
    print('Training on %d samples, %d epochs with batch size %d...' %
          (len(characters), epochs_for_gan, batch_size))

    # Set start time
    start_time = datetime.datetime.now()

    # Output size of D, here 8 == 128 / 2 ** 4
    gan_patch = (8, 8, 1)

    # Define ground truth for D
    target_truth = np.ones((batch_size,) + gan_patch)
    fake_truth = np.zeros((batch_size,) + gan_patch)

    # Tarin D first and G next for each epoch
    for epoch_i in range(epochs_for_gan):
        data = generate_training_data(
            font, characters, target_images, batch_size)

        d_loss_total = .0
        g_loss_total = .0

        for (raw_image, target_image) in data:
            # Train D
            fake_image = generator.predict(raw_image)
            discriminator.train_on_batch(
                [target_image, raw_image], target_truth)
            discriminator.train_on_batch(
                [fake_image, raw_image], fake_truth)

            # Train G (now D.trainable == false)
            gan_loss = gan.train_on_batch(
                [target_image, raw_image], [target_truth, target_image])

            d_loss_total += gan_loss[1]
            g_loss_total += gan_loss[2]

        duration = datetime.datetime.now() - start_time
        start_time = datetime.datetime.now()

        # Print epoch & loss
        print('epoch %d/%d \t D loss: %f, G loss: %f \t time: %ds, ETA: %s' %
              (epoch_i+1, epochs_for_gan, d_loss_total/batch_size, g_loss_total/batch_size, duration.seconds, (epochs_for_gan-epoch_i-1)*duration))

        # Save image & model
        if (epoch_i + 1) % save_image_rate == 0:
            save_image(epoch_i)
        if (epoch_i + 1) % save_model_rate == 0:
            save_model()


train()
print(colorama.Fore.GREEN + colorama.Style.BRIGHT + 'Train finished.')

image_mosaick()
print(colorama.Fore.GREEN + colorama.Style.BRIGHT +
      'Successfully output mosaic image.')
