from keras.layers import Input, Reshape, Dropout, Concatenate, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from non_local import non_local_block
from PIL import Image, ImageDraw, ImageFont
import datetime
import os
import numpy as np


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define abspaths
fonts_dir = 'raw_fonts'
target_img_dir = 'target_img'
fake_img_dir = 'fake_img'
model_data_dir = 'model_data/pretraining.h5'


# Define hyperparameters
epochs_for_gan = 200
batch_size = 32
save_model_rate = 50
save_image_rate = 10
learning_rate = 0.05


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
    gf = 32

    def conv2d(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same')(layer_input)
        d = PReLU()(d)
        if bn:
            d = BatchNormalization(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                   padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(u)
        u = Concatenate()([u, skip_input])
        return u

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
    output_img = Conv2D(1, kernel_size=4,
                        strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)


# Define PatchGAN discriminator
def build_discriminator():
    df = 32

    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same')(layer_input)
        d = PReLU()(d)
        if bn:
            d = BatchNormalization(d)
        return d

    img_A = Input(shape=(128, 128, 1))
    img_B = Input(shape=(128, 128, 1))

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)


# Complie GAN model
discriminator = build_discriminator()
discriminator.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['accuracy'])
discriminator.trainable = False
generator = build_generator()

img_A = Input(shape=(128, 128, 1))
img_B = Input(shape=(128, 128, 1))
fake_A = self.generator(img_B)
valid = self.discriminator([fake_A, img_B])

gan = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
gan.compile(loss=['mse', 'mae'],
            loss_weights=[1, 100],
            optimizer=Adam(lr=learning_rate))


# Dynamically generate training data
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
                X = np.array(X)
                X = X.reshape(X.shape[0], 128, 128, 1).astype(
                    'float32') / 127.5 - 1
                yield (X, np.array(Y))
                X = []
                Y = []


# Save sample images
def save_image():
    print('Saving fake images.')
    fake_images = gan.predict(x=batch_raw_images, verbose=1)
    for character, fake_image in zip(batch_characters, fake_images):
        save_image = ((fake_image + 1) *
                      127.5).astype('uint8').reshape(128, 128)
        Image.fromarray(save_image, mode='L').save(
            '{}/{}pre{}.png'.format(fake_img_dir, character, epoch + 1))


# Save model
def save_model():
    print('Saving model...')
    gan.save(model_data_dir)
    print('Saving model succeeded.')


# Train GAN
gan_patch = (64, 64)

def train():
    # Set start time
    start_time = datetime.datetime.now()

    # Define ground truth for D
    real = np.ones((batch_size,) + gan_patch)
    fake = np.zeros((batch_size,) + gan_patch)

    # Tarin D first and G next for each epoch
    for epoch in range(epochs_for_gan):
        for batch_i, (raw_image, target_image) in enumerate(generate_training_data(font, characters, target_images, batch_size)):
            # Train D
            fake_image = generator.predict(raw_image)
            d_loss_target = discriminator.train_on_batch([raw_image, target_image], real)
            d_loss_fake = discriminator.train_on_batch([fake_image, target_image], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train G (now D.trainable == false)
            g_loss = gan.train_on_batch([raw_image, target_image], [real, raw_image])

            # Print epoch & loss
            print("--- epochs %d/%d --- D loss = %f --- G loss = %f --- time: %s", epoch + 1, epochs_for_gan, d_loss, g_loss, datetime.datetime.now() - start_time)
            start_time = datetime.datetime.now()

            # Save image & model
            if (epoch + 1) % save_image_rate == 0:
                save_image()
            if (epoch + 1) % save_model_rate == 0:
                save_model()

train()
