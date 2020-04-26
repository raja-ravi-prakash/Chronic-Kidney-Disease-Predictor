# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:37:47 2020

@author: USER
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
app = Flask(__name__)
model = pickle.load(open('decisionp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('indexp.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[int(x) for x in request.form.values()]]
    print(x_test)
    sc = load('scalarp.save') 
    prediction = model.predict(sc.transform(x_test))
    print(prediction)
    output=prediction[0]
    if(output==0):
        pred="not cdk"
    else:
        pred="cdk"
    
    return render_template('indexp.html', prediction_text='User {}'.format(pred))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
