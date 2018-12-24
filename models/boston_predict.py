import pandas as pd
from sklearn.externals import joblib

### WORKING
#model = joblib.load('/home/pedro/repos/ml_web_api/model-deployment-flask/models/model.pkl')
#X_test = pd.read_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/data/body_mass/X_test.csv", encoding='utf-8', sep=",")
#res = model.predict(X_test)


model = joblib.load('/home/pedro/repos/ml_web_api/model-deployment-flask/models/pipe_model_boston.pkl')
X_test = pd.read_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/data/boston_housing/X_test.csv", encoding='utf-8', sep=",")
res = model.predict(X_test)


print(res)