from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
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

print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))

# start Grid search
parameters = {'C':[0.001, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
log_reg = LogisticRegression(multi_class='auto')
log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
log_reg_grid_results =  fm.perform_model(log_reg_grid, X_train, y_train, X_test, y_test, class_labels=labels)

plt.figure(figsize=(8,8))
plt.grid(b=False)
fm.confusion_matrix(log_reg_grid_results['confusion_matrix'], classes=labels, cmap=plt.cm.Greens, )
plt.show()

# observe the attributes of the model
fm.search_attributes(log_reg_grid_results['model'])