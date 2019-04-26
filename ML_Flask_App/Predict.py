# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import librosa
import pandas as pd
import numpy as np

import pathlib
import os
#import timeit
from pandas import DataFrame

def predict(song):
    #load models
    model_bilstm = pickle.load(open("model-bilstm.pkl",'rb'))
    model_keras = pickle.load(open("model-keras.pkl",'rb'))
    model_keras2 = pickle.load(open("model-keras2.pkl",'rb'))
    model_lstm = pickle.load(open("model-lstm.pkl",'rb')) 

#get filename
    y, sr = librosa.load(song, mono=True, duration=30)

    rmse = librosa.feature.rmse(y=y)
    predict = []

# new features
    flatness = librosa.feature.spectral_flatness(y=y)
    S = np.abs(librosa.stft(y))
    p0 = librosa.feature.poly_features(S=S, order=0)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr) 
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(flatness)} {np.mean(p0)} {np.mean(chroma_cq)} {np.mean(chroma_cens)} {np.mean(contrast)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    for i in tonnetz:
        to_append += f' {np.mean(i)}'
    predict = to_append.split()

    data = np.array(predict)

    test = data.reshape((1, 1, 37))
    data_keras = data.reshape((1,37))

    genres_dict = {"0" : "blues", "1" : "classical","2" : "country", "3" : "disco","4" : "hiphop", "5" : "jazz","6" : "metal", "7" : "pop","8" : "reggae", "9" : "rock"}

    result_lstm = model_lstm.predict(test)
    result_bilstm = model_bilstm.predict(test)
    result_keras = model_keras.predict(data_keras)
    result_keras2 = model_keras2.predict(data_keras)

    predict_lstm = np.argmax(result_lstm)
    predict_bilstm = np.argmax(result_bilstm)
    predict_keras = np.argmax(result_keras)
    predict_keras2 = np.argmax(result_keras2)

    print('Result for LSTM:',(result_lstm, genres_dict.get(str(predict_lstm))))
    print('Result for BiLSTM:',(result_bilstm, genres_dict.get(str(predict_bilstm))))
    print('Result for Keras:',(result_keras, genres_dict.get(str(predict_keras))))
    print('Result for Keras2:',(result_keras2, genres_dict.get(str(predict_keras2))))
    print(song)
    return ((result_lstm, genres_dict.get(str(predict_lstm))), (result_bilstm, genres_dict.get(str(predict_bilstm))), (result_keras, genres_dict.get(str(predict_keras))), (result_keras2, genres_dict.get(str(predict_keras2))) )

if __name__ == '__main__' :
    
    #song = "Yung_Kartz_-_04_-_Hold_Up.au" #files.upload()
    genre = predict(song)
    print(genre)