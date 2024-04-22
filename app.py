from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)

data=pd.read_csv('Social_Network_Ads.csv')
with open("model.pkl","rb") as model_file:
    model=pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    gender=request.form['Gender']
    
    age=int(request.form['Age'])
    salary=float(request.form['Salary'])
    features=([[age,salary]])
    prediction=model.predict(features)
    target=prediction[0]
    if target==1:
        result="Yes,It is a purchase"
    else:
        result="No,No purchase"
    return render_template('index.html',pred_result=result)
    
if __name__=="__main__":
    app.run(debug=True)

