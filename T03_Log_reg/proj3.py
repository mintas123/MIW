import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

plt.style.use('seaborn-whitegrid')
pd.set_option('display.max_columns', 500)


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # training stage
    def fit(self, X, y):
        # generate random weights
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.cost_ = []

        # on a set of data
        for i in range(self.n_iter):
            net_input = self.net_input(X)  # inputs * weights + bias

            outputs = self.activation(net_input)  # activation function -> probability on a set of data, φ(z)

            errors = (y - outputs)
            self.update_weights(X, errors)

            cost = self.cost_fn(y, outputs)  # minimization of the cost function
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        # multiply inputs and weights + bias . It is called a score.
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        # find probabilities in the range of [0, 1]
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        # predict the class given features
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # decision boundary, threshold

    def cost_fn(self, labels, outputs):
        '''
        labels: numpy array, shape is (7000, )
        outputs: numpy array, shape is (7000, )
        '''
        return -labels.dot(np.log(outputs)) - ((1 - labels).dot(np.log(1 - outputs)))

    def update_weights(self, X, errors):
        self.w_[1:] += self.eta * X.T.dot(errors)
        self.w_[0] += self.eta * errors.sum()


def prepare_data(filename):
    data = pd.read_csv(filename, header=0)
    data = data.dropna()
    data['education'] = np.where(data['education'] == ' Preschool', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == ' 1st-4th', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == ' 5th-6th', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == ' 7th-8th', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == ' 9th', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == ' 10th', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == ' 11th', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == ' 12th', 'Basic', data['education'])

    print(data.groupby('Prediction').mean())
    return data


def main():
    test_data = prepare_data('adult.data')
    train_data = prepare_data('adult.test')

    lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd.fit(X_train_01_subset, y_train_01_subset)
    plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
