import io
import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
#import numpy as np
#from scipy import misc
import pandas as pd

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		#file = request.files['image']
		#if not file: return render_template('index.html', label="No file")
		
		# read in file as raw pixels values
		# (ignore extra alpha channel and reshape as its a single image)
		#img = misc.imread(file)
		#img = img[:,:,:3]
		#img = img.reshape(1, -1)

		# make prediction on new image
		data_file = request.files['dataset']
		data = data_file.read()
		dat = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		print("%"*60)
		print(dat)
		print("%"*60)
		print(type(dat))
		print("-------------")
		#print(data_file.seek(0))

		prediction = model.predict(dat)
		prediction_output = pd.DataFrame(prediction).reset_index(drop=False)
		prediction_output.columns = ["ID", "y_hat"]
		prediction_output.to_csv("prediction.csv", index=False)
		print(prediction_output.head())


		#print("prediction: ",pd.DataFrame(prediction).to_string())
	
		# squeeze value from 1D array and convert to string for clean return
		#label = str(np.squeeze(prediction))

		# switch for case where label=10 and number=0
		#if label=='10': label='0'

		return render_template('index.html', label=pd.DataFrame(prediction).to_string())


if __name__ == '__main__':
	# load ml model
	model = joblib.load('model.pkl')
	# start api
	app.run(host='0.0.0.0', port=8000, debug=True)
