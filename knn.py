#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing


# import pylab as pl

def main(data, size_n_neighbors, metric_distance):
    # loads data
    print("Loading data...")
    X_data, y_data = load_svmlight_file(data)
    # splits data
    print("Spliting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=5)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # fazer a normalizacao dos dados #######
    # scaler = preprocessing.MinMaxScaler()
    # X_train = scaler.fit_transform(X_train_dense)
    # X_test = scaler.fit_transform(X_test_dense)

    # cria um kNN
    print('Using n_neighbors='+str(size_n_neighbors))
    print('Using metric='+metric_distance)
    neigh = KNeighborsClassifier(n_neighbors=size_n_neighbors, metric=metric_distance)

    print('Fitting knn')
    neigh.fit(X_train, y_train)

    # predicao do classificador
    print('Predicting...')
    y_pred = neigh.predict(X_test)

    # mostra o resultado do classificador na base de teste
    print('Accuracy: ', neigh.score(X_test, y_test))

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


if __name__ == "__main__":

    avaiable_metrics = {'EU': 'euclidean', 'MA': 'manhattan', 'CH': 'chebyshev', 'MI': 'minkowski'}

    if len(sys.argv) < 1:
        sys.exit("Use: knn.py <data> [n_neighbors]")

    n_neighbors = 1
    if len(sys.argv) > 2:
        n_neighbors = int(sys.argv[2])

    metric_distance = 'euclidean'
    if len(sys.argv) > 3:
        metric_distance = avaiable_metrics[str(sys.argv[3])]

    main(sys.argv[1], n_neighbors, metric_distance)
