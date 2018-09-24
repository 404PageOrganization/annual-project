from keras.utils import np_utils
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, PReLU
from PIL import Image
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
raw_img_dir = 'raw_img'
real_img_dir = 'real_img'
fake_img_dir = 'fake_img'
model_data_dir = 'model_data'
data_file_name = 'model.pickle'


# Define running args
run_epochs = 5
epochs_for_generator = 100
save_model_rate = 5


# Define model args
l2_rate = 0.01


# Load model datas
trained = os.path.exists(model_data_dir + os.sep + data_file_name)

if trained:
    model_data = open(model_data_dir + os.sep + data_file_name, 'rb')
    start_epoch = pickle.load(model_data)
    generatorr_initial_epoch = pickle.load(model_data)
    model_data.close()
else:
    start_epoch = 0
    generatorr_initial_epoch = 0
    discriminator_initial_epoch = 0


# Read real images & characters
real_images = []
characters = []

for real_img_file in [name for name in os.listdir(real_img_dir) if name[0] != '.']:
    for file_name in [name for name in os.listdir(real_img_dir + os.sep + real_img_file) if name[0] != '.']:
        real_images.append(list(Image.open(real_img_dir + os.sep +
                                           real_img_file + os.sep + file_name).getdata()))
        characters.append(real_img_file)


# Read raw images & characters
raw_images = []

for raw_img_file in characters:
    for file_name in [name for name in os.listdir(raw_img_dir + os.sep + raw_img_file) if name[0] != '.']:
        raw_images.append(list(Image.open(raw_img_dir + os.sep +
                                          raw_img_file + os.sep + file_name).getdata()))


# Process image
raw_images = numpy.array(raw_images)
raw_images = raw_images.reshape(
    raw_images.shape[0], 128, 128, 2).astype('float32') / 127.5 - 1
real_images = numpy.array(real_images)
real_images = real_images.reshape(
    real_images.shape[0], 128, 128, 2).astype('float32') / 127.5 - 1


# Save sample images
real_sample = ((real_images[0] + 1) * 127.5).astype('uint8')
raw_sample = ((raw_images[0] + 1) * 127.5).astype('uint8')
Image.fromarray(real_sample, mode='LA').save(
    fake_img_dir + os.sep + 'real_img.png')
Image.fromarray(raw_sample, mode='LA').save(
    fake_img_dir + os.sep + 'raw_img.png')

# Define the models
generator = Sequential([
    Conv2D(input_shape=(128, 128, 2),
           filters=8,
           kernel_size=3,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=32,
           kernel_size=3,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=64,
           kernel_size=5,
           strides=2,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    BatchNormalization(),
    Conv2D(filters=64,
           kernel_size=5,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=32,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=8,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same'),
    PReLU(),
    UpSampling2D(size=2),
    BatchNormalization(),
    Conv2D(filters=2,
           kernel_size=3,
           # kernel_initializer='uniform',
           kernel_regularizer=l2(l2_rate),
           padding='same',
           activation='tanh'),
])


# Print model struct
print(generator.summary())

# Load trained models
if trained:
    generator = load_model(model_data_dir + os.sep + 'generator.h5')

# Compile models
generator.compile(loss='logcosh',
                  optimizer='Adadelta',
                  metrics=['acc'])

# Train models
for epoch in range(start_epoch + 1, start_epoch + run_epochs + 1):
    print(Fore.BLUE + Style.BRIGHT + 'Epoch:{}'.format(epoch))

    length = len(characters)

    # Training generator

    generator.fit(x=raw_images,
                  y=real_images,
                  initial_epoch=generatorr_initial_epoch,
                  epochs=generatorr_initial_epoch + epochs_for_generator,
                  verbose=2)

    generatorr_initial_epoch += epochs_for_generator

    # Save fake images
    print(Fore.BLUE + Style.BRIGHT + 'Generating fake images.')
    fake_images = generator.predict(x=raw_images, verbose=1)

    for character, fake_image in zip(characters, fake_images):
        save_image = ((fake_image + 1) * 127.5).astype('uint8')
        Image.fromarray(save_image, mode='LA').save(
            fake_img_dir + os.sep + character + str(epoch) + '.png')

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
