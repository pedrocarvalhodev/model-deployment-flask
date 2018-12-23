import io
import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
#import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':
		# 1. Get data from request 
		data_file = request.files['dataset']
		data = data_file.read()
		dat = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")

		# 2. Predict and save results
		prediction = model.predict(dat)
		prediction_output = pd.DataFrame(prediction).reset_index(drop=False)
		prediction_output.columns = ["ID", "y_hat"]
		prediction_output.to_csv("prediction.csv", index=False)
		print(prediction_output.head())



		return render_template('index.html', label="Prediction processed. Check folder for results.")


if __name__ == '__main__':
	# load ml model
	model = joblib.load('model.pkl')
	# start api
	app.run(host='0.0.0.0', port=8000, debug=True)
