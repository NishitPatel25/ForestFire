import os
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import pickle
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

## import ridge regressor model and standard scalar pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## route for homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        #we need to read the value, what user will write on app
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain =float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html') 

print('Model loaded. Check http://127.0.0.1:5000/')



if __name__ == '__main__':
    app.run(debug=True)