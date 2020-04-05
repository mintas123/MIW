import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

plt.style.use('seaborn-whitegrid')


def plot_dataset(f, l):
    # Females
    x0 = f[l == 0, 0]
    y0 = f[l == 0, 1]
    plt.plot(x0, y0, 'ok', label='all Females')

    # Males
    x1 = f[l == 1, 0]
    y1 = f[l == 1, 1]
    plt.plot(x1, y1, 'ob', label='all Males')


def plot_query(q, correct):
    x = q[0]
    y = q[1]

    if correct:
        plt.plot(x, y, 'og')
    else:
        plt.plot(x, y, 'or')


def load_data():
    raw_data = []
    # used the same dataset for making sure its working fine, if it is a problem I can change it
    with open('weight-height.csv', 'r') as file:
        for line in file.readlines():
            data_row = line.strip().split(',')
            raw_data.append(data_row)
    return np.array(raw_data)


def manhattan_distance(point1, point2):
    return sum(abs(a - b) for a, b in zip(point1, point2))


def inch_to_cm(value):
    return float(value) * 2.54


def lb_to_kg(value):
    return float(value) * 0.45359237


def preprocess(raw_data):
    le = LabelEncoder()
    label_rows = []

    features = []
    labels = []

    for row in raw_data:
        feature_row = list(map(float, [inch_to_cm(row[1]), lb_to_kg(row[2])]))
        # I changed the imperials to metric just to make it more clear visually
        label = row[0]
        features.append(feature_row)
        label_rows.append(label)
    labels = le.fit_transform(label_rows)

    return np.array(features), np.array(labels)


def knn(features, labels, query, k, distance_fn):
    # I didnt change knn function as it was already made by You, or I could use some imported tool
    # but it wasnt the main point of the project right?
    all_neighbors = []
    for index, sample in enumerate(features):
        distance = distance_fn(sample, query)
        all_neighbors.append((distance, index))

    sorted_all_neighbors = sorted(all_neighbors)
    k_neighbor_distances = sorted_all_neighbors[:k]

    k_labels = np.array([labels[index] for _, index in k_neighbor_distances])
    k_neighbors = np.array([features[index] for _, index in k_neighbor_distances])

    return k_neighbors, k_labels, mode(k_labels)


def main():
    data = load_data()
    plt.xlabel('height [cm]')
    plt.ylabel('weight [kg]')
    k = 5

    features, labels = preprocess(data)
    l_train, l_test, f_train, f_test = train_test_split(labels, features, test_size=0.05)
    plot_dataset(f=f_train, l=l_train)  # test_size is 5% as it takes quite a while to show graph

    aM_pM = 0   # actual Male predicted Male etc.
    aM_pF = 0
    aF_pF = 0
    aF_pM = 0

    for q in range(len(f_test)):
        correct = False
        query = f_test[q]

        k_neighbors, k_classes, predicted_class = knn(f_train, l_train, query, k, manhattan_distance)

        if predicted_class[0] == l_test[q]:  # if good guess
            correct = True
            if predicted_class[0] == 1:     # if guessed Male
                aM_pM += 1
            else:
                aF_pF += 1
        else:                               # if bad guess
            if predicted_class[0] == 1:
                aF_pM += 1
            else:
                aM_pF += 1
        plot_query(q=query, correct=correct)


    print('         Male:    Female:')
    print(f'Male     {aM_pM}      {aF_pM}')
    print(f'Female   {aM_pF}       {aF_pF}')
    print(f'Overall correctness: {(aM_pM+aF_pF)/len(f_test)}')

    plt.legend()
    plt.show()


main()
