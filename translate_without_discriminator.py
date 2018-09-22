from keras.utils import np_utils
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, PReLU, ELU
from PIL import Image, ImageDraw, ImageFont
from colorama import init, Fore, Style
import pickle
import os
import numpy


# See https://stackoverflow.com/questions/42270739/how-do-i-resolve-these-tensorflow-warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Init colorama
init(autoreset=True)


# Define abspaths
fonts_dir = 'fonts'
real_img_dir = 'real_img'
fake_img_dir = 'fake_img'
model_data_dir = 'model_data'
data_file_name = 'model.pickle'


# Define running args
run_epochs = 10
epochs_for_generator = 5
save_image_rate = 1
save_model_rate = 5


# Define model args
l2_rate = 0.01

# Load model datas
trained = os.path.exists(model_data_dir + os.sep + data_file_name)

if trained:
    model_data = open(model_data_dir + os.sep + data_file_name, 'rb')
    start_epoch = pickle.load(model_data)
    generatorr_initial_epoch = pickle.load(model_data)
    discriminator_initial_epoch = pickle.load(model_data)
    model_data.close()
else:
    start_epoch = 0
    generatorr_initial_epoch = 0
    discriminator_initial_epoch = 0


# Read real images & characters
real_images = []
characters = []

for real_img_file in [name for name in os.listdir(real_img_dir) if name != '.DS_Store']:
    for file_name in [name for name in os.listdir(real_img_dir + os.sep + real_img_file) if name != '.DS_Store']:
        real_images.append(list(Image.open(real_img_dir + os.sep +
                                           real_img_file + os.sep + file_name).getdata()))
        characters.append(real_img_file)


# Generate raw images
raw_images = []

# Read 1 font
font_name = [name for name in os.listdir(fonts_dir) if name != '.DS_Store'][0]

# Read font by using truetype
font = ImageFont.truetype(fonts_dir + os.sep + font_name, 96)

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
    raw_images.shape[0], 128, 128, 2).astype('float32') / 255
real_images = numpy.array(real_images)
real_images = real_images.reshape(
    real_images.shape[0], 128, 128, 2).astype('float32') / 255


# Define the models
generator = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=12,
           kernel_size=3,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=24,
           kernel_size=3,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=48,
           kernel_size=3,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Flatten(),
    Dense(units=12288,
          # kernel_initializer='uniform',
          kernel_regularizer=l2(l2_rate)),
    PReLU(),
    BatchNormalization(),
    Dense(units=2048,
          # kernel_initializer='uniform',
          kernel_regularizer=l2(l2_rate)),
    PReLU(),
    BatchNormalization(),
    Dense(units=128,
          # kernel_initializer='uniform',
          kernel_regularizer=l2(l2_rate)),
    PReLU(),
    BatchNormalization(),
    Dense(units=128,
          # kernel_initializer='uniform',
          kernel_regularizer=l2(l2_rate)),
    PReLU(),
    BatchNormalization(),
    Dense(units=2048,
          # kernel_initializer='uniform',
          kernel_regularizer=l2(l2_rate)),
    PReLU(),
    BatchNormalization(),
    Dense(units=12288,
          # kernel_initializer='uniform',
          kernel_regularizer=l2(l2_rate)),
    Reshape((16, 16, 48)),
    UpSampling2D(size=2),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=24,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    UpSampling2D(size=2),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=12,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    UpSampling2D(size=2),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=2,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same',
           activation='sigmoid'),
])


# Print model struct
print(generator.summary())


# Compile models
generator.compile(loss='logcosh',
                  optimizer='sgd',
                  metrics=['acc'])

# Train models
for epoch in range(start_epoch + 1, start_epoch + run_epochs + 1):
    print(Fore.BLUE + Style.BRIGHT + 'Epoch:{}'.format(epoch))

    length = len(characters)

    # Training generator
    print(Fore.BLUE + Style.BRIGHT + 'Training generator.')

    generator.fit(x=raw_images,
                  y=real_images,
                  initial_epoch=generatorr_initial_epoch,
                  epochs=generatorr_initial_epoch + epochs_for_generator,
                  verbose=2)

    # Save image
    if(epoch % save_image_rate == 0):
        # Generating fake images
        print(Fore.BLUE + Style.BRIGHT + 'Generating fake images.')
        fake_images = generator.predict(x=raw_images,
                                        verbose=1)
        save_image = (fake_images[0] * 255).astype('uint8')
        Image.fromarray(save_image, mode='LA').save(
            fake_img_dir + os.sep + str(epoch) + '.png')

        print(Fore.GREEN + Style.BRIGHT + 'Image saved.')

    # Save models
    if(epoch % save_model_rate == 0):
        # Write now epoch
        model_data = open(model_data_dir + os.sep + data_file_name, 'wb')
        pickle.dump(epoch, model_data)
        pickle.dump(generatorr_initial_epoch, model_data)
        model_data.close()

        # Save model
        generator.save(model_data_dir + os.sep + 'generator.h5')

        print(Fore.GREEN + Style.BRIGHT + 'Models saved.')
