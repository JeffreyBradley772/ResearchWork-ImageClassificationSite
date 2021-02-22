from skimage.color import rgb2gray
from skimage import io
from skimage.feature import hog
from skimage.transform import rescale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import skimage
from skimage.transform import resize
from os import listdir
from os.path import isfile, join
from imblearn.over_sampling import *
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as Pipe
from joblib import dump, load
import sys


# Much of the code for HOG preprocessing comes from: https://kapernikov.com/tutorial-image-classification-with-scikit-learn/


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X], dtype=object)


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X], dtype=object)
        except:
            return np.array([local_hog(img) for img in X], dtype=object)


def show_image(img):
    """Displays an image (input needs to be an array or name of file)"""
    io.imshow(img)
    io.show()


def read_image(filename):
    """Reads an image with a filename"""
    return io.imread(filename)


def get_pipe():
    """Returns the preprocessing pipeline"""
    # create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(8, 8),
        cells_per_block=(4, 4),
        orientations=9,
        block_norm='L2'
    )
    scalify = load('scalify.joblib')

    pipeline = Pipe([('grayify', RGB2GrayTransformer()),
                     ('hogify', hogify),
                     ('scalify', scalify)])
    return pipeline


def hog_example(filename):
    """Visualization of the HOG transoformation for one image"""

    # read the image and resize it to 64x64 pixels
    rat = resize(read_image(filename), (64, 64))

    # calculate the hog and return a visual representation.
    rat_hog, rat_hog_img = hog(
        rat, pixels_per_cell=(8, 8),
        cells_per_block=(4, 4),
        orientations=9,
        visualize=True,
        block_norm='L2')

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 6)
    # remove ticks and their labels
    [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for a in ax]

    ax[0].imshow(rat, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(rat_hog_img, cmap='gray')
    ax[1].set_title('After HOG')
    # plt.savefig("CL_HOG.png")
    # plt.savefig("Cysts_HOG.png")
    plt.show()


def get_image(path):
    """Given the path of an image, returns the image as a numpy array, resized at 64x64 pixels"""
    return resize(read_image(path), (64, 64))


def get_xandy(mypath, num):
    """Adds all of the images in a directory to the X array and the label ("num" parameter) to the y array"""

    # Read all of the images in the directory and add them to X
    X = [resize(read_image(mypath+"/"+f), (64, 64))
         for f in listdir(mypath) if isfile(join(mypath, f))]

    # For each image in the directory, add a corresponding label (specified by num) to the y array
    y = [num for i in range(len(X))]

    return X, y


def read_data():
    """Reads and returns the preprocessed datasets from saved csvs"""

    print("Reading data from CSV files...")
    return np.genfromtxt('PreprocessedData/x_train.csv'), np.genfromtxt('PreprocessedData/x_test.csv'), np.genfromtxt('PreprocessedData/y_train.csv'), np.genfromtxt('PreprocessedData/y_test.csv')


def preprocess_data(save=False):
    """Preprocesses the raw data and saves it to csvs if specified by the save flag"""

    print("Fetching and processing data...")

    X = []
    y = []

    # Read the image files as np arrays and resize them so each image is the same size (64x64 pixels for now)
    # add the images to our X dataset, and then add the labels to the y dataset (0 for CL, 1 for Cysts, 2 for Follicles, 3 for GF)
    print("Reading raw data")
    for i in set(list(range(1, 31))) - set([5, 29]):
        print("Rat", i)

        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_CL", 0)
        X += X_temp
        y += y_temp
        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_Cysts", 1)
        X += X_temp
        y += y_temp
        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_Follicles", 2)
        X += X_temp
        y += y_temp
        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_GF", 3)
        X += X_temp
        y += y_temp
    print("Train test split")

    # split data into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    # create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(8, 8),
        cells_per_block=(4, 4),
        orientations=9,
        block_norm='L2'
    )
    scalify = StandardScaler()

    # Create the preprocessing pipeline
    pipeline = Pipe([('grayify', grayify),
                     ('hogify', hogify),
                     ('scaler', scalify)])

    # Run our X_train and X_test dataset through the preprocessing pipeline
    print("Processing data through pipeline: testing set")
    X_test_prepared = pipeline.fit_transform(X_test, y_test)
    print("Processing data through pipeline: training set")
    X_train_prepared = pipeline.fit_transform(X_train, y_train)

    # Creating an oversampling instance using SVMSMOTE to create synthetic data for Cysts and GF
    over = SVMSMOTE(random_state=42, sampling_strategy='minority',
                    svm_estimator=SVC(kernel='sigmoid'))

    # Perform two rounds of oversampling on the training set so that both Cysts and GF are oversampled
    # Note: We only oversample the training set because we do not want to test on synthetic data, we only want to train on it

    print("Oversampling pt 1")
    # transform the dataset
    X_train_prepared, y_train = over.fit_resample(
        X_train_prepared, y_train)
    # print("Oversampling pt2")
    # X_train_prepared, y_train = over.fit_resample(
    #     X_train_prepared, y_train)

    # If the save parameter is set to true, save the X_train, X_test, y_train, and y_test datasets to csvs
    if save:
        print("Saving data to csvs")
        np.savetxt("x_train.csv", X_train_prepared)
        np.savetxt("x_test.csv", X_test_prepared)
        np.savetxt("y_train.csv", y_train)
        np.savetxt("y_test.csv", y_test)
        print("Saved data to csvs")

    # return the relevant datasets
    return X_train_prepared, X_test_prepared, y_train, y_test


def save_stdsc():
    """Creates, fits, and saves a standard scaler object"""

    print("Fetching and processing data...")

    X = []
    y = []

    # Read the image files as np arrays and resize them so each image is the same size (64x64 pixels for now)
    # add the images to our X dataset, and then add the labels to the y dataset (0 for CL, 1 for Cysts, 2 for Follicles, 3 for GF)
    print("Reading raw data")
    for i in set(list(range(1, 31))) - set([5, 29]):
        print("Rat", i)

        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_CL", 0)
        X += X_temp
        y += y_temp
        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_Cysts", 1)
        X += X_temp
        y += y_temp
        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_Follicles", 2)
        X += X_temp
        y += y_temp
        X_temp, y_temp = get_xandy(
            "Rat " + str(i).zfill(2) + "/Rat" + str(i).zfill(2) + "_GF", 3)
        X += X_temp
        y += y_temp

    print("Train test split")

    # split data into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    # create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(8, 8),
        cells_per_block=(4, 4),
        orientations=9,
        block_norm='L2'
    )
    scalify = StandardScaler()

    # Create preprocessing pipeline without the standard scaler
    pipeline = Pipe([('grayify', grayify),
                     ('hogify', hogify)])

    # Run the data through the first two steps of the pipeline
    print("Processing data through pipeline: training set")
    X_train_prepared = pipeline.fit_transform(X_train, y_train)

    # Fit and save the standard scaler based on the preprocessed X_train dataset
    scalify = scalify.fit(X_train_prepared)
    dump(scalify, 'scalify.joblib')


if __name__ == "__main__":

    #hog_example('Rat 19/Rat19_CL/rat19_01_CL01.tif')
    #hog_example('Rat 19/Rat19_Cysts/rat19_04_CYST01.tif')
    preprocess_data(save=False)
    # save_stdsc()
