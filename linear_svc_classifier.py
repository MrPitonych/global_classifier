from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import data_loader as dl
import full_method as fm
import warnings

train = dl.data_loading('./train.csv')
test = dl.data_loading('./test.csv')
labels = dl.labels_loading()

# get X_train and y_train from csv files
X_train = train.drop(['Activity', 'ActivityName'], axis=1)
y_train = train.Activity

# get X_test and y_test from test csv file
X_test = test.drop(['Activity', 'ActivityName'], axis=1)
y_test = test.Activity


parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
lr_svc = LinearSVC(tol=0.00005)
lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
lr_svc_grid_results = fm.perform_model(lr_svc_grid, X_train, y_train, X_test, y_test, class_labels=labels)

fm.search_attributes(lr_svc_grid_results['model'])