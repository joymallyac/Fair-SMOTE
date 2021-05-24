
# coding: utf-8

# In[1]:


import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics


import sys
sys.path.append(os.path.abspath('..'))

from SMOTE import smote
from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy


# # Load Dataset

# In[2]:


## Load dataset
from sklearn import preprocessing
dataset_orig = pd.read_csv('../data/adult.data.csv')

## Drop NULL values
dataset_orig = dataset_orig.dropna()

## Drop categorical features
dataset_orig = dataset_orig.drop(['workclass','fnlwgt','education','marital-status','occupation','relationship','native-country'],axis=1)

## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)


## Discretize age
dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 60 ) & (dataset_orig['age'] < 70), 60, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 50 ) & (dataset_orig['age'] < 60), 50, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 40 ) & (dataset_orig['age'] < 50), 40, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 30 ) & (dataset_orig['age'] < 40), 30, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 20 ) & (dataset_orig['age'] < 30), 20, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 10 ) & (dataset_orig['age'] < 10), 10, dataset_orig['age'])
dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])

protected_attribute = 'sex'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)


dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)
# dataset_orig


# # Check Original Score

# In[3]:


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']


clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)


print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


# # Check SMOTE Scores

# In[4]:


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)

X_train, y_train = sm.fit_sample(X_train, y_train.ravel())


zero_zero = 0
zero_one = 0
one_zero = 0
one_one = 0

j = 0
for i in X_train:
    if i[3] == 0 and y_train[j] == 0:
        zero_zero += 1
    if i[3] == 0 and y_train[j] == 1:
        zero_one += 1
    if i[3] == 1 and y_train[j] == 0:
        one_zero += 1
    if i[3] == 1 and y_train[j] == 1:
        one_one += 1
    j += 1
        
print(zero_zero , zero_one, one_zero, one_one)
      
      
# --- LSR
clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


# In[5]:


adult_df = dataset_orig

#Based on class
adult_df_one , adult_df_zero = [x for _, x in adult_df.groupby(adult_df['Probability'] == 0)]

#Based on sex
adult_df_one_male, adult_df_one_female = [x for _, x in adult_df_one.groupby(adult_df_one['sex'] == 0)]
adult_df_zero_male, adult_df_zero_female = [x for _, x in adult_df_zero.groupby(adult_df_zero['sex'] == 0)]

#Based on race
adult_df_one_white, adult_df_one_nonwhite = [x for _, x in adult_df_one.groupby(adult_df_one['race'] == 0)]
adult_df_zero_white, adult_df_zero_nonwhite = [x for _, x in adult_df_zero.groupby(adult_df_zero['race'] == 0)]

print(adult_df_one_male.shape,adult_df_one_female.shape,adult_df_zero_male.shape,adult_df_zero_female.shape)
print(adult_df_one_white.shape,adult_df_one_nonwhite.shape,adult_df_zero_white.shape,adult_df_zero_nonwhite.shape)


X_train, y_train = adult_df.loc[:, adult_df.columns != 'Probability'], adult_df['Probability']


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)

X_train, y_train = sm.fit_sample(X_train, y_train.ravel())


zero_zero = 0
zero_one = 0
one_zero = 0
one_one = 0

j = 0
for i in X_train: ## for sex
    if i[3] == 0 and y_train[j] == 0:
        zero_zero += 1
    if i[3] == 0 and y_train[j] == 1:
        zero_one += 1
    if i[3] == 1 and y_train[j] == 0:
        one_zero += 1
    if i[3] == 1 and y_train[j] == 1:
        one_one += 1
    j += 1
        
print(zero_zero , zero_one, one_zero, one_one)


zero_zero = 0
zero_one = 0
one_zero = 0
one_one = 0

j = 0
for i in X_train: ## for race
    if i[2] == 0 and y_train[j] == 0:
        zero_zero += 1
    if i[2] == 0 and y_train[j] == 1:
        zero_one += 1
    if i[2] == 1 and y_train[j] == 0:
        one_zero += 1
    if i[2] == 1 and y_train[j] == 1:
        one_one += 1
    j += 1
        
print(zero_zero , zero_one, one_zero, one_one)


# In[6]:


from plotly import graph_objects as go
import matplotlib.pyplot as plt

data = {
"High & Privileged": [(adult_df_one_male.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                      31433/74310 * 100,
       (adult_df_one_white.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                     33673/74310 * 100],
"High & Unprivileged": [(adult_df_one_female.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                        5722/74310 * 100,
       (adult_df_one_nonwhite.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                       3482/74310 * 100],
"Low & Privileged": [(adult_df_zero_male.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                     22732/74310 * 100,
       (adult_df_zero_white.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                    31155/74310 * 100],
"Low & Unprivileged": [(adult_df_zero_female.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                       14423/74310 * 100,
       (adult_df_zero_nonwhite.shape[0]/(adult_df_one.shape[0] + adult_df_zero.shape[0]) * 100),
                      6000/74310 * 100],
    "labels": [
        "Sex(Original)",
        "Sex(After SMOTE)",
        "Race(Original)",
        "Race(After SMOTE)",
    ]
}

fig = go.Figure(
    data=[
        go.Bar(
            name="High & Privileged",
            x=data["labels"],
            y=data["High & Privileged"],
            offsetgroup=0,            
            textangle=26
        ),
        go.Bar(
            name="High & Unprivileged",
            x=data["labels"],
            y=data["High & Unprivileged"],
            offsetgroup=0,
            base=data["High & Privileged"],            
            textangle=26
        ),
        go.Bar(
            name="Low & Privileged",
            x=data["labels"],
            y=data["Low & Privileged"],
            offsetgroup=1,            
            textangle=26
        ),
        go.Bar(
            name="Low & Unprivileged",
            x=data["labels"],
            y=data["Low & Unprivileged"],
            offsetgroup=1,
            base=data["Low & Privileged"],
            textangle=26
        )
    ],
    layout=go.Layout(
        yaxis_title="% of data points"
    )
)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.75,
    font=dict(
#             family="Courier",
            size=9,
            color="black"
        )
))

fig.write_image("Adult_Imbalance.pdf") 

fig.show()


# # Check RUS scores

# In[7]:


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

zero_zero = 0
zero_one = 0
one_zero = 0
one_one = 0

j = 0
for i in X_train:
    if i[3] == 0 and y_train[j] == 0:
        zero_zero += 1
    if i[3] == 0 and y_train[j] == 1:
        zero_one += 1
    if i[3] == 1 and y_train[j] == 0:
        one_zero += 1
    if i[3] == 1 and y_train[j] == 1:
        one_one += 1
    j += 1
        
print(zero_zero , zero_one, one_zero, one_one)
      
      
# --- LSR
clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


# # Check ROS Scores

# In[8]:


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

zero_zero = 0
zero_one = 0
one_zero = 0
one_one = 0

j = 0
for i in X_train:
    if i[3] == 0 and y_train[j] == 0:
        zero_zero += 1
    if i[3] == 0 and y_train[j] == 1:
        zero_one += 1
    if i[3] == 1 and y_train[j] == 0:
        one_zero += 1
    if i[3] == 1 and y_train[j] == 1:
        one_one += 1
    j += 1
        
print(zero_zero , zero_one, one_zero, one_one)
      
      
# --- LSR
clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))


# # Check KMeans-SMOTE scores

# In[9]:


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

from kmeans_smote import KMeansSMOTE

sm = KMeansSMOTE(random_state=42)

X_train, y_train = sm.fit_sample(X_train, y_train.ravel())


zero_zero = 0
zero_one = 0
one_zero = 0
one_one = 0

j = 0
for i in X_train:
    if i[3] == 0 and y_train[j] == 0:
        zero_zero += 1
    if i[3] == 0 and y_train[j] == 1:
        zero_one += 1
    if i[3] == 1 and y_train[j] == 0:
        one_zero += 1
    if i[3] == 1 and y_train[j] == 1:
        one_one += 1
    j += 1
        
print(zero_zero , zero_one, one_zero, one_one)
      
      
# --- LSR
clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)

print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))

