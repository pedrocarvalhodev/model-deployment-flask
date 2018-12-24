import scipy.io 
import numpy as np
from sklearn.utils import shuffle 
#from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.externals import joblib

import pandas as pd
import sklearn.datasets as data
# load data file as dict object
#train_data = scipy.io.loadmat('extra_32x32.mat') 

df = data.load_diabetes()
df = df.get("data")
df = pd.DataFrame(df)
df.columns = ["Age","Sex","body_mass_index","avg_blood_pressure","S1","S2","S3","S4","S5","S6"]

y_col = "avg_blood_pressure"
X_cols = ["Age","Sex","body_mass_index","S1","S2","S3","S4","S5","S6"]

# extract the images (X) and labels (y) from the dict
X = df[X_cols] 
y = df[y_col] 

# reshape our matrices into 1D vectors and shuffle (still maintaining the index pairings)
#X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T 
#y = y.reshape(y.shape[0],) 
#X, y = shuffle(X, y, random_state=42)

# split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/body_mass/X_train.csv", index=False)
X_test.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/X_test.csv", index=False)
y_train.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/y_train.csv", index=False)
y_test.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/y_test.csv", index=False)

# define classifier and fit to training data
clf = LinearRegression() 
clf.fit(X_train, y_train) 

# save model
joblib.dump(clf, 'model.pkl')
