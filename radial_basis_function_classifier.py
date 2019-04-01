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
X_train = train.drop(['Activity', 'ActivityName'], axis=1)
y_train = train.Activity

# get X_test and y_test from test csv file
X_test = test.drop(['Activity', 'ActivityName'], axis=1)
y_test = test.Activity

parameters = {'C':[2,8,16],\
              'gamma': [ 0.0078125, 0.125, 2]}
rbf_svm = SVC(kernel='rbf')
rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
rbf_svm_grid_results = fm.perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)

fm.search_attributes(rbf_svm_grid_results['model'])