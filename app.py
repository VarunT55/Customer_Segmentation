from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
with open("Customer_Segmentation.pkl","rb") as f:
    model=pickle.load(f)
f.close()

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Annual_income = int(request.form['Annual_Income'])
        Spending_score=int(request.form['Spending_score'])
        
        prediction=model.predict([[Annual_income,Spending_score]])
        output=prediction[0]
        if output==0:
            return render_template('index.html',prediction_texts="Customer with average income and average spending")
        elif output==1:
            return render_template('index.html',prediction_texts="Customer with high income but low spending")
        elif output==2:
            return render_template('index.html',prediction_texts="Customer with low income and low spending")
        elif output==3:
            return render_template('index.html',prediction_texts="Customer with low income and high spending")
        else:
            return render_template('index.html',prediction_texts="Customer with high income and high spending")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

