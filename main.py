import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump
import sys
import pandas as pd


def get_data(base_dir):
    data_file_names = [x for x in os.listdir(base_dir) if x.endswith('.csv')]
    data = {}
    for name in data_file_names:
        path_file = os.path.join(base_dir, name)
        data[name] = pd.read_csv("C:\Users\pramit\PycharmProjects\pythonProject3\Model\Iris.csv")
    return data


def split_data(data, test_size=0.2, random_state=42):
    df = data['iris.csv']
    X = df.drop('variety', axis=1)
    y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=200)
    mod = GridSearchCV(clf, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]})
    mod.fit(X_train, y_train)
    m = mod.best_estimator_
    return m


def save_model(m):
    dump(m, '../Model/irispred.joblib')
    print("Model saved")


if __name__ == "__main__":
    base_dir = r'C:\Users\pramit\PycharmProjects\pythonProject3\Model\Iris.csv'
    data = get_data(base_dir)

    split_data = split_data(data['iris.csv'])

    m = train_model(split_data['X_train'], split_data['y_train'])
    metrics = create_metrics(split_data['X_test'], split_data['y_test'], m)
    save_model(m)
    save_metrics(metrics)


