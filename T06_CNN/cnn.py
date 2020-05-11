import keras, matplotlib, numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.datasets.mnist as minst
import keras.utils as utils
import matplotlib.pyplot as plt

# load MNIST dataset split into train/test datasets
(X_train, y_train), (X_test, y_test) = minst.load_data(path="mnist.npz")

# show the first image in the dataset (it's already greyscale!)
plt.imshow(X_train[0])
print(y_train[0])

# see the shape of an image
print(f'Shape: {X_train[0].shape}')

# # reshape X train and X test
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# print(f'Shape: {X_train[0].shape}')

# convert labels to binary class matrix. use to_categorical()
y_train = utils.to_categorical(y_train, num_classes=None, dtype="float32")
y_test = utils.to_categorical(y_test, num_classes=None, dtype="float32")
print(y_train[0])

# build a model
# sequential
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# . . . follow the PF for the rest of the structure

# compile with an optimizer, loss, and metrics
model.compile(loss='categorical_crossentropy',
              optimizer="adadelta",
              metrics=['accuracy'])

# fit the training data in the model
model.fit(X_train, y_train,
          epochs=10,
          batch_size=1,
          verbose=1
          # validation_data=(X_test, y_test)
          )

# evaluate your model on the test datasets and show what you've got
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show summary
model.summary()
