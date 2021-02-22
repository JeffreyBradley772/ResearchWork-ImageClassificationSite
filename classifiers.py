from preprocessing import read_data, preprocess_data, get_pipe, get_image
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import skimage
from sklearn.svm import SVC
from confusion_matrix import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import sys
from os import listdir
from os.path import isfile, join
import os
# Get the relevant datasets using the function defined in preprocessing.py
#X_train_prepared, X_test_prepared, y_train, y_test = read_data()
#X_train_prepared, X_test_prepared, y_train, y_test = preprocess_data(save=True)


def train_and_save(clf, filename='classifier.joblib'):
    print("Training the model")
    clf.fit(X_train_prepared, y_train)
    print("Saving the model")
    dump(clf, filename)
    print("Done")


def test_classifier(clf, filename=None):
    """Takes a classifier in as an input and tests it on our dataset"""
    #X_train_prepared, X_test_prepared, y_train, y_test = read_data()
    #X_train_prepared, X_test_prepared, y_train, y_test = preprocess_data()
    X_train_prepared, X_test_prepared, y_train, y_test = preprocess_data_v2()


    print("Training the model")
    clf.fit(X_train_prepared, y_train)
    #clf = load('classifier.joblib')
    #X_test_prepared, y_test = np.genfromtxt('PreprocessedData/x_test.csv'), np.genfromtxt('PreprocessedData/y_test.csv')
    print("Making predictions on the testing set")
    # Make predictions on the testing set and calculates metrics for model performance
    y_pred = clf.predict(X_test_prepared)
    print("Accuracy: %.2f" % (accuracy_score(y_pred, y_test)))
    print("Precision: %.2f" %
          (precision_score(y_pred, y_test, average='weighted')))
    print("Recall: %.2f" % (recall_score(y_pred, y_test, average='weighted')))
    print("F1-Score: %.2f" % (f1_score(y_pred, y_test, average='weighted')))

    print("Creating confusion matrix...")
    # Create confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(2)))
    np.set_printoptions(precision=2)
    print("Plotting confusion matrix...")
    # Plot confusion matrix and save if flag is specified
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[
                          'CL', 'Cysts'])
    if filename:
        plt.savefig(filename)
    plt.show()


def classify_structure(path, clf):
    pred_dict = {0: 'CL', 1: 'Cyst', 2: 'Follicle', 3: 'GF'}
    print("Loading saved classifier")
    #clf = load(clf_filename)
    print("Processing image and making prediction")
    img = []
    img.append(get_image(path))
    pipeline = get_pipe()
    pipeline.steps.append(("classifier", clf))
    prediction = int(pipeline.predict(img))
    print("Prediction for %s: %s" % (path, pred_dict[prediction]))
    return prediction


def classify_batch(path, clf_filename='classifier.joblib'):
    clf = load(clf_filename)
    pred_dict = {0: 'CL', 1: 'Cyst', 2: 'Follicle', 3: 'GF'}
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    for f in filenames:
        prediction = classify_structure(path + "/" + f, clf)
        new_path = 'Labeled Data/%s/%s' % (pred_dict[prediction], f)
        os.rename(path + "/" + f, new_path)


def gridsearch(clf, param_grid):

    # Grid search to determine which hyper parameters are best
    print("Searching for the best parameters...")
    gs = GridSearchCV(estimator=clf(),
                      param_grid=param_grid,
                      scoring='accuracy',
                      refit=True,
                      cv=5,
                      n_jobs=-1)

    gs = gs.fit(X_train_prepared, y_train)
    print("Accuracy:", gs.best_score_)
    print("Parameters:", gs.best_params_)

    gs = GridSearchCV(estimator=clf(),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=2)

    scores = cross_val_score(gs, X_train_prepared, y_train,
                             scoring='accuracy', cv=5)

    print("\n\nCV Accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))


if __name__ == "__main__":

    svm = SVC(gamma=.0001, C=1, kernel='sigmoid')
    param_range = [0.0001, 0.001, 0.01, 0.1,
                   1.0, 10.0, 100.0, 1000.0]

    param_grid = [
        {'C': param_range,
         'gamma': param_range,
         'kernel': ['sigmoid']
         }
    ]

    # gridsearch(SVC, param_grid)
    # sys.exit()
    test_classifier(SVC())
    # train_and_save(svm)
    #clf = load('classifier.joblib')
    #classify_structure('Rat 10/Rat10_CL/rat10_02_CL02.tif', clf)
    # filenames = [f for f in listdir(
    #     'Rat 10/Rat10_Cysts') if isfile(join('Rat 10/Rat10_Cysts', f))]
    # for f in filenames:
    #     classify_structure('Rat 10/Rat10_Cysts/' + f)
    #classify_batch('Unlabeled Data/JR2015_rat55')
