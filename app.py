from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__,template_folder='Web')

@app.route('/')
def home():
    result =''
    return render_template('index.html')

@app.route('/',methods=['POST','GET'])
def result():
    if request.method == 'POST':
        ChestTightness = float(request.form['ChestTightness'])
        ChestPain = float(request.form['ChestPain'])
        Smart = float(request.form['Smart'])
        HeartPalpitations = float(request.form['HeartPalpitations'])
        Squeamish = float(request.form['Squeamish'])
        Faint = float(request.form['Faint'])
        Gasp = float(request.form['Gasp'])
        Tired = float(request.form['Tired'])
        Choking = float(request.form['Choking'])
        EpigastricCongestion = float(request.form['EpigastricCongestion'])
        loaded_model= joblib.load('iris-IschemicHeartDisease.sav')
        result = loaded_model.predict([[ChestTightness, ChestPain, Smart, HeartPalpitations, Squeamish, Faint, Gasp, Tired, Choking, EpigastricCongestion]])[0]
    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)