
# coding: utf-8

# ### Analytics Vidhya: Practice Problem (Approach)

# In[2]:


import os
import re
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# In[3]:


#!ls /home/pratos/Side-Project/av_articles/data/


# Download the training & test data from the Practice Problem approach. We'll do a bit of quick investigation on the dataset:

# In[4]:


path="/home/pedro/repos/ml_web_api/flask_api"


# In[5]:


data = pd.read_csv(path+'/data/training.csv')


# In[6]:


data.head()


# In[7]:


print("Shape of the data is:{}".format(data.shape))


# In[8]:


print("List of columns is: {}".format(list(data.columns)))


# Here, `Loan_status` is our `target variable`, the rest are `predictor variables`. `Loan_ID` wouldn't help much in making predictions about `defaulters` hence we won't be considering that variable in our final model.

# Finding out the `null/Nan` values in the columns:

# In[9]:


for _ in data.columns:
    print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))


# We'll check out the values (labels) for the columns having missing values:

# In[10]:


missing_pred = ['Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Gender', 'Married']

for _ in missing_pred:
    print("List of unique labels for {}:::{}".format(_, set(data[_])))


# For the rest of the missing values:
# 
# - `Dependents`: Assumption that there are no dependents
# - `Self_Employed`: Assumption that the applicant is not self-employed
# - `Loan_Amount_Term`: Assumption that the loan amount term is median value
# - `Credit_History`: Assumption that the person has a credit history
# - `Married`: If nothing specified, applicant is not married
# - `Gender`: Assuming the gender is Male for the missing values
# 
# Before that we'll divide the dataset in train and test

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


list(data.columns)


# In[13]:


pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'],                                                     test_size=0.25, random_state=42)


# We'll compile a list of `pre-processing` steps that we do on to create a custom `estimator`.

# In[15]:


X_train['Dependents'] = X_train['Dependents'].fillna('0')
X_train['Self_Employed'] = X_train['Self_Employed'].fillna('No')
X_train['Loan_Amount_Term'] = X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mean())


# In[16]:


X_train['Credit_History'] = X_train['Credit_History'].fillna(1)
X_train['Married'] = X_train['Married'].fillna('No')
X_train['Gender'] = X_train['Gender'].fillna('Male')


# In[17]:


X_train['LoanAmount'] = X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())


# We have a lot of `string` labels that we encounter in `Gender`, `Married`, `Education`, `Self_Employed` & `Property_Area` columns.

# In[18]:


label_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']

for _ in label_columns:
    print("List of unique labels {}:{}".format(_, set(X_train[_])))


# In[19]:


gender_values = {'Female' : 0, 'Male' : 1} 
married_values = {'No' : 0, 'Yes' : 1}
education_values = {'Graduate' : 0, 'Not Graduate' : 1}
employed_values = {'No' : 0, 'Yes' : 1}
property_values = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
X_train.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values,                 'Self_Employed': employed_values, 'Property_Area': property_values, 'Dependents': dependent_values}                , inplace=True)


# In[20]:


X_train.head()


# In[21]:


X_train.dtypes


# In[22]:


for _ in X_train.columns:
    print("The number of null values in:{} == {}".format(_, X_train[_].isnull().sum()))


# Converting the pandas dataframes to numpy arrays:

# In[23]:


X_train = X_train.as_matrix()


# In[24]:


X_train.shape


# We'll create a custom `pre-processing estimator` that would help us in writing better pipelines and in future deployments:

# In[25]:


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
        pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',            'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
        
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


# To make sure that this works, let's do a test run for it:

# In[26]:


X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'],                                                     test_size=0.25, random_state=42)


# In[27]:


X_train.head()


# In[28]:


for _ in X_train.columns:
    print("The number of null values in:{} == {}".format(_, X_train[_].isnull().sum()))


# In[29]:


preprocess = PreProcessing()


# In[30]:


preprocess


# In[31]:


preprocess.fit(X_train)


# In[32]:


X_train_transformed = preprocess.transform(X_train)


# In[33]:


X_train_transformed.shape


# So our small experiment to write a custom `estimator` worked. This would be helpful further.

# In[34]:


X_test_transformed = preprocess.transform(X_test)


# In[35]:


X_test_transformed.shape


# In[36]:


y_test = y_test.replace({'Y':1, 'N':0}).as_matrix()


# In[37]:


y_train = y_train.replace({'Y':1, 'N':0}).as_matrix()


# In[38]:


param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
             "randomforestclassifier__max_depth" : [None, 6, 8, 10],
             "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
             "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}


# In[39]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(PreProcessing(),
                    RandomForestClassifier())


# In[40]:


pipe


# In[41]:


from sklearn.model_selection import train_test_split, GridSearchCV

grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)


# In[42]:


grid


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'],                                                     test_size=0.25, random_state=42)


# In[44]:


grid.fit(X_train, y_train)


# In[45]:


print("Best parameters: {}".format(grid.best_params_))


# In[46]:


print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))

