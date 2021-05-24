
# coding: utf-8

# In[ ]:


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
from Generate_Samples import generate_samples


# # Load Dataset

# In[ ]:


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

protected_attribute1 = 'sex'
protected_attribute2 = 'race'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)


dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle = True)
# dataset_orig


# # Check Original Score - Sex

# In[ ]:


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR

print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'F1'))
print("aod :"+protected_attribute1,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'aod'))
print("eod :"+protected_attribute1,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'DI'))


# # Check Original Score - Race

# In[ ]:


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR

print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'F1'))
print("aod :"+protected_attribute2,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'aod'))
print("eod :"+protected_attribute2,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'DI'))


# # Find Class & Protected attribute Distribution

# In[ ]:


# first one is class value and second one is 'sex' and third one is 'race'
zero_zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 0)])
zero_zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 1)])
zero_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 0)])
zero_one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 1)])
one_zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 0)])
one_zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 1)])
one_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 0)])
one_one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 1)])


print(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)


# # Sort these four

# In[ ]:


maximum = max(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)
if maximum == zero_zero_zero:
    print("zero_zero_zero is maximum")
if maximum == zero_zero_one:
    print("zero_zero_one is maximum")
if maximum == zero_one_zero:
    print("zero_one_zero is maximum")
if maximum == zero_one_one:
    print("zero_one_one is maximum")
if maximum == one_zero_zero:
    print("one_zero_zero is maximum")
if maximum == one_zero_one:
    print("one_zero_one is maximum")
if maximum == one_one_zero:
    print("one_one_zero is maximum")
if maximum == one_one_one:
    print("one_one_one is maximum")


# In[ ]:


zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
zero_zero_one_to_be_incresed = maximum - zero_zero_one
zero_one_zero_to_be_incresed = maximum - zero_one_zero
zero_one_one_to_be_incresed = maximum - zero_one_one
one_zero_zero_to_be_incresed = maximum - one_zero_zero
one_zero_one_to_be_incresed = maximum - one_zero_one
one_one_zero_to_be_incresed = maximum - one_one_zero
one_one_one_to_be_incresed = maximum - one_one_one

print(zero_zero_zero_to_be_incresed,zero_zero_one_to_be_incresed,zero_one_zero_to_be_incresed,zero_one_one_to_be_incresed,
     one_zero_zero_to_be_incresed,one_zero_one_to_be_incresed,one_one_zero_to_be_incresed,one_one_one_to_be_incresed)


# In[ ]:


df_zero_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 0)]
df_zero_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 1)]
df_zero_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 0)]
df_zero_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 1)]
df_one_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 0)]
df_one_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 1)]
df_one_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 0)]
df_one_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                       & (dataset_orig_train[protected_attribute2] == 1)]


df_zero_zero_zero['race'] = df_zero_zero_zero['race'].astype(str)
df_zero_zero_zero['sex'] = df_zero_zero_zero['sex'].astype(str)

df_zero_zero_one['race'] = df_zero_zero_one['race'].astype(str)
df_zero_zero_one['sex'] = df_zero_zero_one['sex'].astype(str)

df_zero_one_zero['race'] = df_zero_one_zero['race'].astype(str)
df_zero_one_zero['sex'] = df_zero_one_zero['sex'].astype(str)

df_zero_one_one['race'] = df_zero_one_one['race'].astype(str)
df_zero_one_one['sex'] = df_zero_one_one['sex'].astype(str)

df_one_zero_zero['race'] = df_one_zero_zero['race'].astype(str)
df_one_zero_zero['sex'] = df_one_zero_zero['sex'].astype(str)

df_one_zero_one['race'] = df_one_zero_one['race'].astype(str)
df_one_zero_one['sex'] = df_one_zero_one['sex'].astype(str)

df_one_one_zero['race'] = df_one_one_zero['race'].astype(str)
df_one_one_zero['sex'] = df_one_one_zero['sex'].astype(str)

df_one_one_one['race'] = df_one_one_one['race'].astype(str)
df_one_one_one['sex'] = df_one_one_one['sex'].astype(str)


df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_incresed,df_zero_zero_zero,'Adult')
df_zero_zero_one = generate_samples(zero_zero_one_to_be_incresed,df_zero_zero_one,'Adult')
df_zero_one_zero = generate_samples(zero_one_zero_to_be_incresed,df_zero_one_zero,'Adult')
df_zero_one_one = generate_samples(zero_one_one_to_be_incresed,df_zero_one_one,'Adult')
df_one_zero_zero = generate_samples(one_zero_zero_to_be_incresed,df_one_zero_zero,'Adult')
df_one_zero_one = generate_samples(one_zero_one_to_be_incresed,df_one_zero_one,'Adult')
df_one_one_zero = generate_samples(one_one_zero_to_be_incresed,df_one_one_zero,'Adult')
df_one_one_one = generate_samples(one_one_one_to_be_incresed,df_one_one_one,'Adult')


# # Append the dataframes

# In[ ]:


df = pd.concat([df_zero_zero_zero,df_zero_zero_one,df_zero_one_zero,df_zero_one_one,
df_one_zero_zero,df_one_zero_one,df_one_one_zero,df_one_one_one])
df


# # Check Score after oversampling

# In[ ]:


X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR


print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'F1'))
print("aod :"+protected_attribute1,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'aod'))
print("eod :"+protected_attribute1,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute1, 'DI'))

print("-------------")

print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'recall'))
print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'far'))
print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'precision'))
print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'accuracy'))
print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'F1'))
print("aod :"+protected_attribute2,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'aod'))
print("eod :"+protected_attribute2,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'eod'))

print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'SPD'))
print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute2, 'DI'))


# # Verification 

# In[ ]:


zero_zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute1] == '0.0')
                                       & (df[protected_attribute2] == '0.0')])
zero_zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute1] == '0.0')
                                       & (df[protected_attribute2] == '1.0')])
zero_one_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute1] == '1.0')
                                       & (df[protected_attribute2] == '0.0')])
zero_one_one = len(df[(df['Probability'] == 0) & (df[protected_attribute1] == '1.0')
                                       & (df[protected_attribute2] == '1.0')])
one_zero_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute1] == '0.0')
                                       & (df[protected_attribute2] == '0.0')])
one_zero_one = len(df[(df['Probability'] == 1) & (df[protected_attribute1] == '0.0')
                                       & (df[protected_attribute2] == '1.0')])
one_one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute1] == '1.0')
                                       & (df[protected_attribute2] == '0.0')])
one_one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute1] == '1.0')
                                       & (df[protected_attribute2] == '1.0')])


print(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)

