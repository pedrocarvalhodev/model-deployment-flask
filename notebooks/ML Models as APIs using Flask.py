
# coding: utf-8

# ## Machine Learning models as APIs using Flask
# 
# ### Introduction
# 
# A lot of Machine Learning (ML) projects, amateur and professional, start with an aplomb. The early excitement with working on the dataset, answering the obvious & not so obvious questions & presenting the results are what everyone of us works for. There are compliments thrown around and talks about going to the next step -- that's when the question arises, __How?__
# 
# The usual suspects are making dashboards and providing insights. But mostly, the real use of your Machine Learning model lies in being at the heart of a product -- that maybe a small component of an automated mailer system or a chatbot. These are the times when the barriers seem unsurmountable. Giving an example, majority of ML folks use `R/Python` for their experiments. But consumer of those ML models would be software engineers who use a completely different stack. There are two ways via which this problem can be solved:
# 
# - __Rewriting the whole code in the language that the software engineering folks work__
# 
# The above seems like a good idea, but the time & energy required to get those intricate models replicated would be utterly waste. Majority of languages like `JavaScript`, do not have great libraries to perform ML. One would be wise to stay away from it.
# 
# - __API-first approach__
# 
# Web APIs have made it easy for cross-language applications to work well. If a frontend developer needs to use your ML Model to create a ML powered web application, he would just need to get the `URL Endpoint` from where the API is being served. 
# 
# The articles below would help you to appreciate why APIs are a popular choice amongst developers:
# 
# - [History of APIs](http://apievangelist.com/2012/12/20/history-of-apis/)
# - [Introduction to APIs - AV Article](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-apis-application-programming-interfaces-5-apis-a-data-scientist-must-know/)
# 
# Majority of the Big Cloud providers and smaller Machine Learning focussed companies provide ready-to-use APIs. They cater to the needs of developers/businesses that don't have expertise in ML, who want to implement ML in their processes or product suites.
# 
# One such example of Web APIs offered is the [Google Vision API](https://cloud.google.com/vision/)
# 
# ![Google API Suite](http://www.publickey1.jp/2016/gcpnext16.jpg)
# 
# All you need is a simple REST call to the API via SDKs (Software Development Kits) provided by Google. [Click here](https://github.com/GoogleCloudPlatform/cloud-vision/tree/master/python) to get an idea of what can be done using Google Vision API.
# 
# Sounds marvellous right! In this article, we'll understand how to create our own Machine Learning API using `Flask`, a web framework with `Python`. 
# 
# ![Flask](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Flask_logo.svg/640px-Flask_logo.svg.png)
# 
# __NOTE:__ `Flask` isn't the only web-framework available. There's `Django`, `Falcon`, `Hug` and many more. For `R`, we have a package called [`plumber`](https://github.com/trestletech/plumber).
# 
# ### Table of Contents:
# 
# 1. __Python Environment Setup & Flask Basics__
# 2. __Creating a Machine Learning Model__
# 3. __Saving the Machine Learning Model: Serialization & Deserialization__
# 4. __Creating an API using Flask__
# 
# ### 1. Python Environment Setup & Flask Basics
# 
# ![Anaconda](https://upload.wikimedia.org/wikipedia/en/c/cd/Anaconda_Logo.png)
# 
# - Creating a virtual environment using `Anaconda`. If you need to create your workflows in Python and keep the dependencies separated out or share the environment settings, `Anaconda` distributions are a great option. 
#     * You'll find a miniconda installation for Python [here](https://conda.io/miniconda.html)
#     * `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
#     * `bash Miniconda3-latest-Linux-x86_64.sh`
#     * Follow the sequence of questions.
#     * `source .bashrc`
#     * If you run: `conda`, you should be able to get the list of commands & help.
#     * To create a new environment, run: `conda create --name <environment-name> python=3.6`
#     * Follow the steps & once done run: `source activate <environment-name>`
#     * Install the python packages you need, the two important are: `flask` & `gunicorn`.
#     
#     
# - We'll try out a simple `Flask` Hello-World application and serve it using `gunicorn`:
# 
#     * Open up your favourite text editor and create `hello-world.py` file in a folder
#     * Write the code below:
#         ```python
# 
#         """Filename: hello-world.py
#         """
# 
#         from flask import Flask
# 
#         app = Flask(__name__)
# 
#         @app.route('/users/<string:username>')
#         def hello_world(username=None):
# 
#             return("Hello {}!".format(username))
# 
#         ```
#     * Save the file and return to the terminal.
#     * To serve the API (to start running it), execute: `gunicorn --bind 0.0.0.0:8000 hello-world:app` on your terminal.
#     
#     * If you get the repsonses below, you are on the right track:
# 
#     ![Hello World](https://raw.githubusercontent.com/pratos/flask_api/master/notebooks/images/flaskapp1.png)
# 
#     * On you browser, try out: `https://localhost:8000/users/any-name`
# 
#     ![Browser](https://raw.githubusercontent.com/pratos/flask_api/master/notebooks/images/flaskapp2.png)
# 
# Viola! You wrote your first Flask application. As you have now experienced with a few simple steps, we were able to create web-endpoints that can be accessed locally. And it remains simple going forward too.
# 
# Using `Flask`, we can wrap our Machine Learning models and serve them as Web APIs easily. Also, if we want to create more complex web applications (that includes JavaScript `*gasps*`) we just need a few modifications.

# ### 2. Creating a Machine Learning Model
# 
# - We'll be taking up the Machine Learning competition: [Loan Prediction Competition](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii). The main objective is to set a pre-processing pipeline and creating ML Models with goal towards making the ML Predictions easy while deployments. 
# 
# 

# In[1]:


import os 
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")


# - Saving the datasets in a folder:

# In[2]:


#!ls /home/pratos/Side-Project/av_articles/flask_api/data/
path="/home/pedro/repos/ml_web_api/flask_api"


# In[3]:


data = pd.read_csv(path+'/data/training.csv')


# In[4]:


list(data.columns)


# In[5]:


data.shape


# - Finding out the `null/Nan` values in the columns:

# In[6]:


for _ in data.columns:
    print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))


# - Next step is creating `training` and `testing` datasets:

# In[7]:


pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'],                                                     test_size=0.25, random_state=42)


# - To make sure that the `pre-processing steps` are followed religiously even after we are done with experimenting and we do not miss them while predictions, we'll create a __custom pre-processing Scikit-learn `estimator`__.
# 
# __To follow the process on how we ended up with this `estimator`, read up on [this notebook](https://github.com/pratos/flask_api/blob/master/notebooks/AnalyticsVidhya%20Article%20-%20ML%20Model%20approach.ipynb)__

# In[8]:


from sklearn.base import BaseEstimator, TransformerMixin

class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """

    def __init__(self):
        pass

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """
        pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',                    'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
        
        df = df[pred_var]
        
        df['Dependents'] = df['Dependents'].fillna(0)
        df['Self_Employed'] = df['Self_Employed'].fillna('No')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(self.term_mean_)
        df['Credit_History'] = df['Credit_History'].fillna(1)
        df['Married'] = df['Married'].fillna('No')
        df['Gender'] = df['Gender'].fillna('Male')
        df['LoanAmount'] = df['LoanAmount'].fillna(self.amt_mean_)
        
        gender_values = {'Female' : 0, 'Male' : 1} 
        married_values = {'No' : 0, 'Yes' : 1}
        education_values = {'Graduate' : 0, 'Not Graduate' : 1}
        employed_values = {'No' : 0, 'Yes' : 1}
        property_values = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
        dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
        df.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values,                     'Self_Employed': employed_values, 'Property_Area': property_values,                     'Dependents': dependent_values}, inplace=True)
        
        return df.as_matrix()

    def fit(self, df, y=None, **fit_params):
        """Fitting the Training dataset & calculating the required values from train
           e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in
                transformation of X_test
        """
        
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self


# - Convert `y_train` & `y_test` to `np.array`:

# In[9]:


y_train = y_train.replace({'Y':1, 'N':0}).as_matrix()
y_test = y_test.replace({'Y':1, 'N':0}).as_matrix()


# We'll create a `pipeline` to make sure that all the preprocessing steps that we do are just a single `scikit-learn estimator`.

# In[10]:


pipe = make_pipeline(PreProcessing(),
                    RandomForestClassifier())


# In[11]:


pipe


# To search for the best `hyper-parameters` (`degree` for `PolynomialFeatures` & `alpha` for `Ridge`), we'll do a `Grid Search`:
# 
# - Defining `param_grid`:

# In[12]:


param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
             "randomforestclassifier__max_depth" : [None, 6, 8, 10],
             "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
             "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}


# - Running the `Grid Search`:

# In[13]:


grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)


# - Fitting the training data on the `pipeline estimator`:

# In[14]:


grid.fit(X_train, y_train)


# - Let's see what parameter did the Grid Search select:

# In[15]:


print("Best parameters: {}".format(grid.best_params_))


# - Let's score:

# In[16]:


print("Validation set score: {:.2f}".format(grid.score(X_test, y_test)))


# - Load the test set:

# In[17]:


test_df = pd.read_csv('../data/test.csv', encoding="utf-8-sig")
test_df = test_df.head()


# In[18]:


test_df


# In[19]:


grid.predict(test_df)


# Our `pipeline` is looking pretty swell & fairly decent to go the most important step of the tutorial: __Serialize the Machine Learning Model__

# ### 3. Saving Machine Learning Model : Serialization & Deserialization

# >In computer science, in the context of data storage, serialization is the process of translating data structures or object state into a format that can be stored (for example, in a file or memory buffer, or transmitted across a network connection link) and reconstructed later in the same or another computer environment.
# 
# In Python, `pickling` is a standard way to store objects and retrieve them as their original state. To give a simple example:

# In[20]:


list_to_pickle = [1, 'here', 123, 'walker']

#Pickling the list
import pickle

list_pickle = pickle.dumps(list_to_pickle)


# In[21]:


list_pickle


# When we load the pickle back:

# In[22]:


loaded_pickle = pickle.loads(list_pickle)


# In[23]:


loaded_pickle


# We can save the `pickled object` to a file as well and use it. This method is similar to creating `.rda` files for folks who are familiar with `R Programming`. 
# 
# __NOTE:__ Some people also argue against using `pickle` for serialization[(1)](#no1). `h5py` could also be an alternative.
# 
# We have a custom `Class` that we need to import while running our training, hence we'll be using `dill` module to packup the `estimator Class` with our `grid object`.
# 
# It is advisable to create a separate `training.py` file that contains all the code for training the model ([See here for example](https://github.com/pratos/flask_api/blob/master/flask_api/utils.py)).
# 
# - To install `dill`

# In[1]:


#!pip install dill


# In[25]:


import dill as pickle
#filename = 'model_v1.pk'
filename = 'model_v2.pk'


# In[26]:


with open(path+'/flask_api/models/'+filename, 'wb') as file:
    pickle.dump(grid, file)


# So our model will be saved in the location above. Now that the model `pickled`, creating a `Flask` wrapper around it would be the next step.
# 
# Before that, to be sure that our `pickled` file works fine -- let's load it back and do a prediction:

# In[27]:


with open(path+'/flask_api/models/'+filename ,'rb') as f:
    loaded_model = pickle.load(f)


# In[28]:


loaded_model.predict(test_df)


# Since, we already have the `preprocessing` steps required for the new incoming data present as a part of the `pipeline` we just have to run `predict()`. While working with `scikit-learn`, it is always easy to work with `pipelines`. 
# 
# `Estimators` and `pipelines` save you time and headache, even if the initial implementation seems to be ridiculous. Stich in time, saves nine!

# ### 4. Creating an API using Flask

# We'll keep the folder structure as simple as possible:
# 
# ![Folder Struct](https://raw.githubusercontent.com/pratos/flask_api/master/notebooks/images/flaskapp3.png)
# 
# There are three important parts in constructing our wrapper function, `apicall()`:
# 
# - Getting the `request` data (for which predictions are to be made)
# 
# - Loading our `pickled estimator`
# 
# - `jsonify` our predictions and send the response back with `status code: 200`
# 
# HTTP messages are made of a header and a body. As a standard, majority of the body content sent across are in `json` format. We'll be sending (`POST url-endpoint/`) the incoming data as batch to get predictions.
# 
# (__NOTE:__ You can send plain text, XML, csv or image directly but for the sake of interchangeability of the format, it is advisable to use `json`)

# ```python
# """Filename: server.py
# """
# 
# import os
# import pandas as pd
# from sklearn.externals import joblib
# from flask import Flask, jsonify, request
# 
# app = Flask(__name__)
# 
# @app.route('/predict', methods=['POST'])
# def apicall():
# 	"""API Call
# 	
# 	Pandas dataframe (sent as a payload) from API Call
# 	"""
# 	try:
# 		test_json = request.get_json()
# 		test = pd.read_json(test_json, orient='records')
# 
# 		#To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
# 		test['Dependents'] = [str(x) for x in list(test['Dependents'])]
# 
# 		#Getting the Loan_IDs separated out
# 		loan_ids = test['Loan_ID']
# 
# 	except Exception as e:
# 		raise e
# 	
# 	clf = 'model_v1.pk'
# 	
# 	if test.empty:
# 		return(bad_request())
# 	else:
# 		#Load the saved model
# 		print("Loading the model...")
# 		loaded_model = None
# 		with open('./models/'+clf,'rb') as f:
# 			loaded_model = pickle.load(f)
# 
# 		print("The model has been loaded...doing predictions now...")
# 		predictions = loaded_model.predict(test)
# 		
# 		"""Add the predictions as Series to a new pandas dataframe
# 								OR
# 		   Depending on the use-case, the entire test data appended with the new files
# 		"""
# 		prediction_series = list(pd.Series(predictions))
# 
# 		final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))
# 		
# 		"""We can be as creative in sending the responses.
# 		   But we need to send the response codes as well.
# 		"""
# 		responses = jsonify(predictions=final_predictions.to_json(orient="records"))
# 		responses.status_code = 200
# 
# 		return (responses)
# 
# ```
# 
# Once done, run: `gunicorn --bind 0.0.0.0:8000 server:app`

# Let's generate some prediction data and query the API running locally at `https:0.0.0.0:8000/predict`

# In[29]:


import json
import requests


# In[30]:


"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json',                   'Accept': 'application/json'}

"""Reading test batch
"""
df = pd.read_csv('../data/test.csv', encoding="utf-8-sig")
df = df.head()

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')


# In[31]:


data


# In[33]:


"""POST <url>/predict
"""
## first -> source activate flask_api ->  gunicorn --bind 127.0.0.1:8000 server:app
resp = requests.post("http://0.0.0.0:8000/predict",                     data = json.dumps(data),                    headers= header)


# In[34]:


resp.status_code


# In[35]:


"""The final response we get is as follows:
"""
resp.json()


# ### End Notes

# We have half the battle won here, with a working API that serves predictions in a way where we take one step towards integrating our ML solutions right into our products. This is a very basic API that will help with proto-typing a data product, to make it as fully functional, production ready API a few more additions are required that aren't in the scope of Machine Learning. 
# 
# There are a few things to keep in mind when adopting API-first approach:
# 
# - Creating APIs out of sphagetti code is next to impossible, so approach your Machine Learning workflow as if you need to create a clean, usable API as a deliverable. Will save you a lot of effort to jump hoops later.
# 
# - Try to use version control for models and the API code, `Flask` doesn't provide great support for version control. Saving and keeping track of ML Models is difficult, find out the least messy way that suits you. [This article](https://medium.com/towards-data-science/how-to-version-control-your-machine-learning-task-cad74dce44c4) talks about ways to do it.
# 
# - Specific to `sklearn models` (as done in this article), if you are using custom `estimators` for preprocessing or any other related task make sure you keep the `estimator` and `training code` together so that the model pickled would have the `estimator` class tagged along. 
# 
# Next logical step would be creating a workflow to deploy such APIs out on a small VM. There are various ways to do it and we'll be looking into those in the next article.
# 
# Code & Notebooks for this article: [pratos/flask_api](https://github.com/pratos/flask_api)

# __Sources & Links:__
# 
# [1]. <a id='no1' href="http://www.benfrederickson.com/dont-pickle-your-data/">Don't Pickle your data.</a>
# 
# [2]. <a id='no2' href="http://www.dreisbach.us/articles/building-scikit-learn-compatible-transformers/">Building Scikit Learn compatible transformers.</a>
# 
# [3]. <a id='no2' href="http://flask.pocoo.org/docs/0.10/security/#json-security">Using jsonify in Flask.</a>
# 
# [4]. <a id='no2' href="http://blog.luisrei.com/articles/flaskrest.html">Flask-QuickStart.</a>
