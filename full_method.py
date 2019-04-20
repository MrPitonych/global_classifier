import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
import pandas as pd


plt.rcParams["font.family"] = 'DejaVu Sans'

def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, \
                  print_cm=True, cm_cmap=plt.cm.Greens):
    # to store results at various phases
    results = dict()

    # time at which model starts training
    train_start_time = datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done \n \n')
    train_end_time = datetime.now()
    results['training_time'] = train_end_time - train_start_time
    print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))

    # predict test data
    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred

    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)


    results['accuracy'] = accuracy
    print('*****************************')
    print('Accuracy:', format(accuracy))
    print('*****************************')

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm:
        print('*****************************')
        print('Confusion Matrix: ')
        print('\n {}'.format(cm))
        print('*****************************')

    # plot confusin matrix
    plt.figure(figsize=(8, 8))
    plt.grid(b=False)
    confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap=cm_cmap)
    plt.savefig('abc.png')


    # get classification report
    print('*****************************')
    print('Classifiction Report:')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    print('*****************************')

    # add the trained  model to the results
    results['model'] = model

    return results

def search_attributes(model):
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('*****************************')
    print('Best Estimator:')
    print('\t{}\n'.format(model.best_estimator_))
    print('*****************************')


    # parameters that gave best results while performing grid search
    print('*****************************')
    print('Best parameters:')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))
    print('*****************************')


    #  number of cross validation splits
    print('*****************************')
    print('No of CrossValidation sets')
    print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))
    print('*****************************')


    # Average cross validated score of the best estimator, from the Grid Search
    print('*****************************')
    print('Best Score:')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))
    print('*****************************')
