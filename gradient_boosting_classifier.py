from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
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

param_grid = {'max_depth': np.arange(5,8,1), \
             'n_estimators':np.arange(130,170,10)}
gbdt = GradientBoostingClassifier()
gbdt_grid = GridSearchCV(gbdt, param_grid=param_grid, n_jobs=-1)
gbdt_grid_results = fm.perform_model(gbdt_grid, X_train, y_train, X_test, y_test, class_labels=labels)
fm.search_attributes(gbdt_grid_results['model'])