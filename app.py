import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def results():
    number_courses = float(request.form['number_courses'])
    time_study = float(request.form['time_study'])

    X = np.array([[number_courses,time_study]])
    model = pickle.load(open('model.pkl','rb'))
    Y_predict = model.predict(X)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    app.run(debug = True, port = 1010)