from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
import joblib
import json
import os

port = int(os.environ.get('PORT', 5000))



digits_dict = {
    0 : "Zero",
    1 : "One",
    2 : "Two",
    3 : "Three",
    4 : "Four",
    5 : "Five",
    6 : "Six",
    7 : "Seven",
    8 : "Eight",
    9 : "Nine"

}


app = Flask(__name__)
CORS(app)

cwd = os.getcwd()
model_address = cwd + "\model.sav"
print (model_address)
model = joblib.load(model_address)

@app.route("/", methods=['GET', 'POST'])
def predictor():
    if (request.method == 'GET') :
        return "Digits API"
    elif (request.method == 'POST') :
        data = (request.get_json()["data"])
        array = data[1:len(data)-1].split(",")
        X = [int(x) for x in array]
        X = np.array(X)
        X = X.reshape(14, 14)
        X = np.transpose(X)
        X_scaled_2x = np.kron(X, np.ones((2,2)))
        
        to_predict = X_scaled_2x.reshape((-1, 28*28))
        
        prediction  = model.predict(to_predict)
        to_send = digits_dict[prediction[0]]
        return to_send

@app.route("/favion.ico", methods=['GET'])
def null() :
    return "NULL";

if __name__=="__main__" :
    app.run(host='0.0.0.0', port=port, debug=True)