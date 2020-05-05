from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    raw_data = []

    with open('dane8.txt', 'r') as file:
        for line in file.readlines():
            data_row = line.strip().split(' ')
            raw_data.append(data_row)
    np.random.shuffle(raw_data)
    data = np.array(raw_data)
    data = data.astype(np.float64)

    x = data[:, 0]
    x = np.reshape(x, (len(data), 1))
    y = data[:, 1]
    y = np.reshape(y, (len(data), 1))

    return train_test_split(x, y, test_size=0.33)


def build_and_compile_model():
    model = Sequential([
        Dense(units=50, input_shape=(1,)),
        Activation('relu'),
        Dense(units=45),
        Activation('relu'),
        Dense(units=1),
    ])
    model.compile(loss='mean_squared_error', optimizer='nadam')
    plot_model(model, to_file='miw5_s16604_model.png', show_layer_names=True, show_shapes=True)

    return model


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    model = build_and_compile_model()

    model.fit(x_train, y_train, epochs=666, verbose=0)

    evaluation = model.evaluate(x_test, y_test)
    print(f'{evaluation} loss')

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("miw5_s16604_model.h5")

    predictions = model.predict(x_test)
    plt.scatter(x_test, predictions, c='r')
    plt.scatter(x_test, y_test, c='b')
    plt.savefig('miw5_s16604_result.png')
    plt.show()
