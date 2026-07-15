from flask import Flask, render_template,request
import joblib
import numpy as numpy
app = Flask(__name__)
model = joblib.load("model/crop_model.pkl")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

        return "Recommended Crop: " + prediction[0]

    return render_template('predict.html')
if __name__ == '__main__':
    app.run(debug=True)
    