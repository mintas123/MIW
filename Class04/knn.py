import numpy as np  # for data
from matplotlib import pyplot as plt  # for visualization
import math  # for square root
from sklearn.preprocessing import LabelEncoder  # for labeling data
from scipy.stats import mode  # for voting

plt.style.use('seaborn-whitegrid')


def plot_dataset(f, l):
    # Females
    x0 = f[l == 0, 0]
    y0 = f[l == 0, 1]
    plt.plot(x0, y0, 'ok', label='all Females')

    # Males
    x1 = f[l == 1, 0]
    y1 = f[l == 1, 1]
    plt.plot(x1, y1, 'or', label='all Males')


def plot_neighbors(f, l):
    x = f[:, 0]
    y = f[:, 1]
    k = len(f)

    plt.plot(x, y, 'oy', label=f'{k} neighbors')


def plot_query(q, p):
    x = q[0]
    y = q[1]

    label = 'Male' if p[0] == 1 else 'Female'

    plt.plot(x, y, 'sg', label=f'It\'s a {label}')

    print(f'Height: {x}')
    print(f'Wight: {y}')
    print(f'Class: {label}')


def load_data():
    raw_data = []
    #     with open('500_Person_Gender_Height_Weight_Index.csv', 'r') as file:
    with open('weight-height.csv', 'r') as file:
        # Discard the first line (headings)
        next(file)

        # Read the data into a table
        for line in file.readlines():
            data_row = line.strip().split(',')
            raw_data.append(data_row)

    return np.array(raw_data)


def euclidean_distance(point1, point2):
    sum_squared_distance = 0

    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)


def manhattan_distance(point1, point2):
    pass


def preprocess(raw_data):
    le = LabelEncoder()
    label_rows = []

    features = []
    labels = []

    for row in raw_data:
        feature_row = list(map(float, row[1:3]))

        # select feature data
        #         label = row[0][1:-1]
        label = row[0]

        features.append(feature_row)
        label_rows.append(label)

    # transform categorical data (labels)
    labels = le.fit_transform(label_rows)

    return np.array(features), np.array(labels)


def knn(features, labels, query, k, distance_fn):
    all_neighbors = []

    for index, sample in enumerate(features):
        # 1
        distance = distance_fn(sample, query)
        all_neighbors.append((distance, index))  # it's a tuple!

    # 2
    sorted_all_neighbors = sorted(all_neighbors)

    # 3
    k_neighbor_distances = sorted_all_neighbors[:k]
    #     print(np.array([labels[index] for _, index in k_neighbor_distances]))

    # 4
    k_labels = np.array([labels[index] for _, index in k_neighbor_distances])
    k_neighbors = np.array([features[index] for _, index in k_neighbor_distances])

    return k_neighbors, k_labels, mode(k_labels)  # mode == voting for the class


def main():
    data = load_data()  # how many columns? what are they?
    plt.xlabel('height [inches]')
    plt.ylabel('weight [pounds]')
    k = 10

    features, labels = preprocess(data)  # how do we split features/labels? what are we doing with N/A or '2,1'?
    plot_dataset(f=features, l=labels)

    query = np.array(
        [70, 175]
    )  # what if it's a 2D array with rows as queries?

    k_neighbors, k_classes, predicted_class = knn(features,
                                                  labels,
                                                  query=query,
                                                  k=k,
                                                  distance_fn=euclidean_distance)  # distance

    plot_neighbors(f=k_neighbors, l=k_classes)
    plot_query(q=query, p=predicted_class)

    plt.legend()
    plt.show()


main()
