from flask import Flask, render_template, url_for, request, redirect, flash
import pickle
import librosa
import pandas as pd
import numpy as np
from pandas import DataFrame
import pathlib
from werkzeug.utils import secure_filename
import Predict
from keras.models import Model
import keras.models
from sklearn.externals import joblib
from pickle import dump 
from pickle import load
import os
from werkzeug import secure_filename
app = Flask(__name__)

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("uploads.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'songs/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    return render_template("home.html")
	
@app.route('/songs', methods=['GET', 'POST'])
#@app.route('/predict', methods=['GET', 'POST'])
def get_prediction():
    songs = os.listdir('./songs')
    for i in songs:
	    my_prediction = Predict.predict(i)
	    print(my_prediction)
		
	 
    return render_template('results.html', prediction = my_prediction)

	
if __name__ == "__main__":
	app.run(debug=True)