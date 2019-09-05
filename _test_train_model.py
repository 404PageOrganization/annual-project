from PIL import Image, ImageDraw, ImageFont
from lib.image_mosaicking import image_mosaick
import colorama
import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsnooper


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
df = 64                  # number of D's conv filters
gf = 64                  # number of G's conv filters
norm_scale = 0.07


# -----------
#  LOAD DATA
# -----------


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
    batch_raw_images.shape[0], 1, 128, 128).astype('float32') / 127.5 - 1
target_images = np.array(target_images)
target_images = target_images.reshape(
    target_images.shape[0], 1, 128, 128).astype('float32') / 127.5 - 1

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

# --------------
#  DEFINE MODEL
# --------------


# TODO: Realise SAME PADDING, which is not out-of-the-box in PyTorch
def get_padding(in_size, out_size, kernel_size, stride=1, dilation=1):
    padding = ((out_size - 1) * stride + 1 - in_size +
               dilation * (kernel_size - 1)) // 2
    return padding


# Define U-Net generator
class down(nn.Module):
    def __init__(self, in_channels, out_channels, is_in=False):
        super().__init__()
        if is_in:
            self.down = nn.Sequential(
                # nn.ZeroPad2d(get_padding(in_size, out_size, kernel_size=3)),
                nn.Conv2d(in_channels, out_channels, kernel_size=3),
                nn.PReLU()
            )
        else:
            self.down = nn.Sequential(
                # nn.ZeroPad2d(get_padding(in_size, out_size, kernel_size=3)),
                nn.Conv2d(in_channels, out_channels, kernel_size=3),
                nn.BatchNorm2d(out_channels),
                nn.PReLU()
            )

    def forward(self, x):
        x = self.down(x)
        return x


class up(nn.Module):
    def __init__(self, in_channels, out_channels, is_out=False):
        super().__init__()
        self.is_out = is_out
        if is_out:
            self.up = nn.Sequential(
                nn.Upsample(size=2),
                # nn.ZeroPad2d(get_padding(in_size, out_size, kernel_size=4)),
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(size=2),
                # nn.ZeroPad2d(get_padding(in_size, out_size, kernel_size=4)),
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if self.is_out:
            return x1
        x = torch.cat((x1, x2), dim=1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d1 = down(in_channels, gf, is_in=True)
        self.d2 = down(gf, gf*2)
        self.d3 = down(gf*2, gf*4)
        self.d4 = down(gf*4, gf*8)
        self.d5 = down(gf*8, gf*8)
        self.d6 = down(gf*8, gf*8)
        self.d7 = down(gf*8, gf*8)
        self.u1 = up(gf*8, gf*8)
        self.u2 = up(gf*8, gf*8)
        self.u3 = up(gf*8, gf*8)
        self.u4 = up(gf*8, gf*4)
        self.u5 = up(gf*4, gf*2)
        self.u6 = up(gf*2, gf)
        self.u7 = up(gf, out_channels)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        u1 = self.u1(d7, d6)
        u2 = self.u2(u1, d5)
        u3 = self.u3(u2, d4)
        u4 = self.u4(u3, d3)
        u5 = self.u5(u4, d2)
        u6 = self.u6(u5, d1)
        u7 = self.u7(u6)
        return F.tanh(u7)


'''
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
'''


# Define PatchGAN discriminator
class d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, is_in=False):
        super().__init__()
        if is_in:
            self.d_layer = nn.Sequential(
                # nn.ZeroPad2d(get_padding(in_size, out_size, kernel_size=4, stride=2)),
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
                nn.PReLU()
            )
        else:
            self.d_layer = nn.Sequential(
                # nn.ZeroPad2d(get_padding(in_size, out_size, kernel_size=4, stride=2)),
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.PReLU()
            )

    def forward(self, a, b=None):
        if b is not None:
            x = torch.cat((a, b), 1)
        x = self.d_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d1 = d_layer(in_channels, df, is_in=True)
        self.d2 = d_layer(df, df*2)
        self.d3 = d_layer(df*2, df*4)
        self.d4 = d_layer(df*4, df*8)
        self.d5 = nn.Conv2d(df*8, 1, kernel_size=4)

    def forward(self, a, b):
        x = self.d1(a, b)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return x


'''
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
'''


# Complie GAN model
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
generator = Generator(1, 1)
discriminator = Discriminator(2, 128)

if torch.cuda.is_available():
    mse_loss.cuda()
    mae_loss.cuda()
    generator.cuda()
    discriminator.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


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
            X = X.reshape(X.shape[0], 1, 128, 128).astype(
                'float32') / 127.5 - 1
            '''
            X += np.random.normal(size=(X.shape[0],
                                        128, 128, 1), scale=norm_scale)
            '''
            Y = np.array(Y)

            yield(X, Y)
            X = []
            Y = []
            it += 1
            if it == len(characters) // batch_size:
                return


# Save sample images
def save_image(epoch):
    pass
    '''
    print('Saving fake images...')
    fake_images = generator(batch_raw_images)
    for character, fake_image in zip(batch_characters, fake_images):
        save_image = ((fake_image + 1) *
                      127.5).astype('uint8').reshape(128, 128)
        Image.fromarray(save_image, mode='L').save(
            '{}/{}{}.png'.format(fake_img_dir, character, epoch + 1))
    print(colorama.Fore.GREEN + colorama.Style.BRIGHT +
          'Images saved successfully.')
    '''


# Save model
def save_model():
    print('Saving model...')
    # gan.save(model_data_dir)
    print(colorama.Fore.GREEN + colorama.Style.BRIGHT +
          'Model saved successfully.')


# Train GAN
# @torchsnooper.snoop()
def train():
    # Print information
    print('Training on {} samples, {} epochs with batch size {}...'.format(
        len(characters), epochs_for_gan, batch_size))

    # Set start time
    start_time = datetime.datetime.now()

    # Define ground truth for D
    # Output size of D is 8 == 128 / 2 ** 4
    gan_patch = (1, 8, 8)
    target_truth = Tensor((batch_size,) + gan_patch).fill_(1.0)
    fake_truth = Tensor((batch_size,) + gan_patch).fill_(0.0)

    # Tarin D first and G next for each epoch
    for epoch_i in range(epochs_for_gan):
        data = generate_training_data(
            font, characters, target_images, batch_size)

        d_loss_total = 0.
        g_loss_total = 0.

        for (raw_image, target_image) in data:
            # print(type(raw_image))
            # print(type(target_image))
            raw_image = torch.from_numpy(raw_image)
            target_image = torch.from_numpy(target_image)
            raw_image = raw_image.cuda()
            target_image = target_image.cuda()
            # Train D
            fake_image = generator(raw_image)
            d_loss = mse_loss([target_image, raw_image], target_truth) + \
                mse_loss([fake_image, raw_image], fake_truth)
            d_loss.backward()
            optimizer_D.step()
            d_loss_total += d_loss

            # Train G
            g_loss = mae_loss(fake_image, target_image)
            g_loss.backward()
            optimizer_G.step()
            g_loss_total += g_loss

        # Calculate duration & reset start time
        duration = datetime.datetime.now() - start_time
        start_time = datetime.datetime.now()

        # Print epoch & loss
        print('epoch {}/{}\tD loss: {}, G loss: {}\ttime: {}s, ETA: {}'.format(epoch_i+1, epochs_for_gan,
                                                                               d_loss_total/batch_size, g_loss_total/batch_size, duration.seconds, (epochs_for_gan-epoch_i-1)*duration))

        # Save image & model
        if (epoch_i + 1) % save_image_rate == 0:
            save_image(epoch_i)
        if (epoch_i + 1) % save_model_rate == 0:
            save_model()


if __name__ == '__main__':
    # Train model
    train()
    print(colorama.Fore.GREEN + colorama.Style.BRIGHT + 'Training completed.')
    # Mosaick images
    image_mosaick()
    print(colorama.Fore.GREEN + colorama.Style.BRIGHT +
        'Successfully output mosaicked image.')
