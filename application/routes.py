from application import application
from flask import render_template,request, flash, redirect, url_for, session
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
from datetime import datetime

from classifiers import classify_structure
from preprocessing import get_pipe



global classifications
classifications = {}


clf = load('classifier.joblib')

@application.route("/")
@application.route("/index")
def index():
    print("Here-2")
    return render_template("index.html")



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','tif'])

def allowed_file(filename):
    print(filename)
    print("here-1")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@application.route('/upload', methods = ['POST', 'GET'])
def upload():
    files = request.files.getlist("inputFile")
    

    

    flash('Classifying...')
    session.pop('_flashes', None)
    for f in files:
        print(f,"FILE")
        if not allowed_file(f.filename):
            flash(' Invalid file')
            return redirect(url_for('import1'))

        


        im = resize(io.imread(f), (64, 64))

        
        pred_dict = {0: 'CL', 1: 'Cyst', 2: 'Follicle', 3: 'GF'}

        img = []
        img.append(im)
        pipeline = get_pipe()
        pipeline.steps.append(("classifier", clf))
        prediction = int(pipeline.predict(img))

        now = datetime.now()
        dt_string = now.strftime("%H:%M:%S %m/%d/%Y ")

        classifications[len(classifications)+1] = (f.filename,pred_dict[prediction],dt_string)
        print(classifications)
        data = [{"Pred":pred_dict[prediction]}]
        print(str(pred_dict[prediction]))
    df = pd.DataFrame.from_dict(classifications, orient='index', columns=['File Name','Classification','Time'])
    return render_template('class.html', data = df.to_html(classes='table table-striped'))
    

@application.route('/import', methods = ['POST', 'GET'])
def import1():
    return render_template('upload.html')

@application.route('/clear', methods = ['POST','GET'])
def clear():
    global classifications
    classifications = {}
    df = pd.DataFrame.from_dict(classifications, orient='index', columns=['File Name','Classification','Time'])
    return render_template('class.html', data = df.to_html(classes='table table-striped'))


@application.route('/about')
def about():
    return render_template('about.html')

@application.route('/contact')
def contact():
    return render_template('contact.html')