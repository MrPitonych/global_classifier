from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import GridSearchCV
import numpy as np
import data_loader as dl
import full_method as fm


def dt_classifier_start():
    train = dl.data_loading('./train.csv')
    test = dl.data_loading('./test.csv')
    labels = dl.labels_loading()

    # get X_train and y_train from csv files
    X_train = train.drop(['Activity', 'ActivityName'], axis=1)
    y_train = train.Activity

    # get X_test and y_test from test csv file
    X_test = test.drop(['Activity', 'ActivityName'], axis=1)
    y_test = test.Activity

    parameters = {'max_depth': np.arange(3, 10, 2)}
    dt = DecisionTreeClassifier()
    dt_grid = GridSearchCV(dt, param_grid=parameters, n_jobs=-1)
    dt_grid_results = fm.perform_model(dt_grid, X_train, y_train, X_test, y_test, class_labels=labels)
    fm.search_attributes(dt_grid_results['model'])

