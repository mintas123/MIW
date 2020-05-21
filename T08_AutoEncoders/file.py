from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

(x_train, _), (x_test, _) = mnist.load_data()

# normalization in range 0-1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# reshape
width = height = 28

x_train = np.reshape(x_train, (len(x_train), width, height, 1))
x_test = np.reshape(x_test, (len(x_test), width, height, 1))

# adding noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# clip to be in (low_limit, upper_limit)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

pyplot.figure(figsize=(30, 3))
for i in range(1, 10):
    num_pic = pyplot.subplot(1, 10, i)
    pyplot.imshow(x_train_noisy[i].reshape(width, height))
    num_pic.get_xaxis().set_visible(False)
    num_pic.get_yaxis().set_visible(False)
    pyplot.gray()
pyplot.show()


def build_encoding_layers(input_img):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded


def build_decoding_layers(encoding_layers):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoding_layers)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded


input_img = Input(shape=(width, height, 1))

encoded = build_encoding_layers(input_img)
decoded = build_decoding_layers(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train,
                epochs=3,
                shuffle=True)

encoder = Model(input_img, encoded)

encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = autoencoder.predict(x_test_noisy)


def visualize(images, amount, total_rows, current_row, new_shape, figsize=None):
    for i in range(1, amount):
        if figsize is not None: pyplot.figure(figsize=figsize)
        ax1 = pyplot.subplot(total_rows, amount, i + amount * current_row)
        pyplot.imshow(images[i].reshape(new_shape))
        pyplot.gray()
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)


amount = 10
pyplot.figure(figsize=(20, 6))

# start row
visualize(images=x_test,
          amount=amount,
          total_rows=3,
          current_row=0,
          new_shape=(width, height),
          )
# noisy row
visualize(images=x_test_noisy,
          amount=amount,
          total_rows=3,
          current_row=1,
          new_shape=(width, height)
          )
#  result row
visualize(images=decoded_imgs,
          amount=amount,
          total_rows=3,
          current_row=2,
          new_shape=(width, height)
          )
pyplot.show()
