from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open('pcos2.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def start_app():
    return render_template('home.html')
@app.route('/overview')
def overview():
    return render_template('pcos1.html')
@app.route('/about')
def ml():
    return render_template('ML.html')
@app.route('/basic')
def basic():
    return render_template('Basic.html')
@app.route('/search')
def predict():
    return render_template('Predict1.html')
@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['height']
    data3 = request.form['weight']
    data4 = request.form['cyc']
    data5 = request.form['cycle']
    data6 = request.form['preg']
    data7 = request.form['ibeta']
    data8 = request.form['iibeta']
    data9 = request.form['fsh']
    data10 = request.form['amh']
    data11 = request.form['gain']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]])
    user_input_prediction = arr.astype('float')
    pred = model.predict(user_input_prediction)
    pred = str(pred)
    pred = pred.replace('[','').replace(']','').strip()
    return render_template('Predict1.html', data=pred)
if __name__ == "__main__":
    app.debug = True
    app.run()