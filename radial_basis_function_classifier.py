from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import data_loader as dl
import full_method as fm

train = dl.data_loading('./train.csv')
test = dl.data_loading('./test.csv')
labels = dl.labels_loading()

# get X_train and y_train from csv files
X_train = train.drop(['Activity'], axis=1)
y_train = train.Activity

# get X_test and y_test from test csv file
X_test = test.drop(['Activity'], axis=1)
y_test = test.Activity

Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['linear'], 'C': Cs}]

rbf_svm = SVC(random_state=12, gamma='auto')
rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, cv=5, n_jobs=-1)
rbf_svm_grid_results = fm.perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)

fm.search_attributes(rbf_svm_grid_results['model'])