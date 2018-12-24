import os 
import json
import numpy as np
import pandas as pd
#import dill as pickle

from sklearn.pipeline import Pipeline

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split #, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

#rom sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

X_train.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/X_train.csv", index=False)
X_test.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/X_test.csv", index=False)
y_train.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/y_train.csv", index=False)
y_test.to_csv("/home/pedro/repos/ml_web_api/model-deployment-flask/y_test.csv", index=False)


def build_and_train():

    df = datasets.load_boston()
    data = pd.DataFrame(df.data, columns=df.feature_names)
    data["medianValue"] = df.target
    y = "medianValue"
    X = [x for x in data.columns if x != y]

    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('preproc',PreProcessing()),
        ('clf',DecisionTreeClassifier())
        ])

    pipeline.fit(X_train,y_train)

	#grid.fit(X_train, y_train)

	return(pipeline)


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """

    def __init__(self):
        pass

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """

        print("Before processing", df.columns)
        df_norm = (df - df.mean()) / (df.max() - df.min())
        df_norm = df_norm.apply(lambda x : np.around(x,1))
        df_norm.columns = [x+"_norm" for x in df.columns]
        df = df.merge(df_norm, how='inner', left_index=True, right_index=True)
        print("After processing", df.columns)
        
        return df.as_matrix()

    #def fit(self, df, y=None, **fit_params):
    #    """Fitting the Training dataset & calculating the required values from train
    #       e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in
    #            transformation of X_test
    #    """
    #    
    #    self.term_mean_ = df['Loan_Amount_Term'].mean()
    #    self.amt_mean_ = df['LoanAmount'].mean()
    #   return self

if __name__ == '__main__':
	clf = build_and_train()

	filename = 'pipeline_model.pkl'
	with open('/models/'+filename, 'wb') as file:
		#pickle.dump(model, file)
        joblib.dump(clf, 'model.pkl')