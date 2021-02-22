from application import application
from flask import render_template,request
from PIL import Image
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
from os.path import isfile, join
import os
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

from classifiers import classify_structure
from preprocessing import get_pipe

from base64 import b64encode



clf = load('classifier.joblib')

@application.route("/")
@application.route("/index")
def index():
    print("Here-2")
    return render_template("index.html")



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    print("here-1")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@application.route('/upload', methods = ['POST', 'GET'])
def upload():
    file = request.files['inputFile']
    im = resize(io.imread(file), (64, 64))
    encoded = b64encode(im)
    encoded = encoded.decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s" % (mime, encoded)
    #print(uri)
    
 
    

    pred_dict = {0: 'CL', 1: 'Cyst', 2: 'Follicle', 3: 'GF'}
    print("Loading saved classifier")
    #clf = load(clf_filename)
    print("Processing image and making prediction")
    img = []
    img.append(im)
    pipeline = get_pipe()
    pipeline.steps.append(("classifier", clf))
    prediction = int(pipeline.predict(img))
    #print("Prediction for %s: %s" % (im, pred_dict[prediction]))
    data = [{"Pred":pred_dict[prediction], "Image":uri}]
    print(str(pred_dict[prediction]))
    
    return render_template('class.html', data = data, uri = uri)
    

@application.route('/import', methods = ['POST', 'GET'])
def import1():
    return render_template('upload.html')
    