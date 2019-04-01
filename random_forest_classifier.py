from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
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

params = {'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(3,15,2)}
rfc = RandomForestClassifier()
rfc_grid = GridSearchCV(rfc, param_grid=params, n_jobs=-1)
rfc_grid_results = fm.perform_model(rfc_grid, X_train, y_train, X_test, y_test, class_labels=labels)
fm.search_attributes(rfc_grid_results['model'])