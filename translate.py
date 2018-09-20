from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, LeakyReLU, ELU
from PIL import Image, ImageDraw, ImageFont
import pickle
import os
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define abspaths
fonts_dir = 'fonts'
real_img_dir = 'real_img'
fake_img_dir = 'fake_img'
model_data_dir = 'model_data'
data_file_name = 'model.pickle'

# Define args
run_epochs = 5
epochs_for_discriminator = 3
epochs_for_generator = 6
save_image_rate = 1
save_model_rate = 5

trained = os.path.exists(model_data_dir + os.sep + data_file_name)

if trained:
    model_data = open(model_data_dir + os.sep + data_file_name, 'rb')
    start_epoch = pickle.load(model_data)
    model_data.close()
else:
    start_epoch = 1

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

generator = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=12,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=24,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=48,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=12288,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=2048,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=128,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=128,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=2048,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=12288,
          kernel_initializer='normal',
          activation='relu'),
    Reshape((16, 16, 48),
            input_shape=(12288,)),
    UpSampling2D(2),
    LeakyReLU(alpha=0.3),
    Conv2D(input_shape=(32, 32, 48),
           filters=24,
           kernel_size=3,
           padding='same'),
    UpSampling2D(2),
    LeakyReLU(alpha=0.3),
    Conv2D(filters=12,
           kernel_size=3,
           padding='same'),
    UpSampling2D(2),
    LeakyReLU(alpha=0.3),
    Conv2D(filters=2,
           kernel_size=3,
           padding='same',
           activation='softmax'),
])

discriminator = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=12,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=24,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=48,
           kernel_size=3,
           padding='same'),
    LeakyReLU(alpha=0.3),
    Dropout(0.25),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=2048,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dropout(0.5),
    Dense(units=16,
          kernel_initializer='normal'),
    ELU(alpha=1.0),
    Dense(units=2,
          kernel_initializer='normal',
          activation='softmax')
])

if trained:
    generator = load_model(model_data_dir + os.sep + 'generator.h5')
    discriminator = load_model(model_data_dir + os.sep + 'discriminator.h5')

# Connect generator with discriminator
discriminator.trainable = False
combine = Sequential([generator, discriminator])

# Print model struct
print(generator.summary())
print(discriminator.summary())
print(combine.summary())

# Compile models
generator.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
discriminator.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
combine.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# Train models
for epoch in range(start_epoch, start_epoch + run_epochs):
    print('Epoch:{}'.format(epoch))

    # Generating fake images
    print('Generating fake images.')
    fake_images = generator.predict(raw_images)

    # Training discriminator
    print('Training discriminator.')

    discriminator.trainable = True
    discriminator.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    images = []

    for real, fake in zip(real_images, fake_images):
        images.append(real)
        images.append(fake)

    images = numpy.array(images)

    length = len(characters)

    y = [1, 0] * length
    y = numpy.array(y)
    y = np_utils.to_categorical(y)

    discriminator.fit(x=images,
                      y=y,
                      batch_size=2,
                      epochs=epochs_for_discriminator,
                      verbose=0)

    # Training generator
    print('Training generator.')

    discriminator.trainable = False
    discriminator.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    y = [[0, 1]] * length
    y = numpy.array(y).astype('float32')

    combine.fit(x=raw_images,
                y=y,
                epochs=epochs_for_generator,
                verbose=2)

    # Save image and models
    if(epoch % save_image_rate == 0):
        save_image = (fake_images[0] * 255).astype('uint8')
        Image.fromarray(save_image, mode='LA').save(
            fake_img_dir + os.sep + str(epoch) + '.png')

    if(epoch % save_model_rate == 0):
        # Write now epoch
        model_data = open(model_data_dir + os.sep + data_file_name, 'wb')
        pickle.dump(epoch + 1, model_data)
        model_data.close()

        # Save model
        generator.save(model_data_dir + os.sep + 'generator.h5')
        discriminator.save(model_data_dir + os.sep + 'discriminator.h5')
