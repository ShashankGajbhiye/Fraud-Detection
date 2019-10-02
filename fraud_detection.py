# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:30:25 2019

@author: Shashank
"""

"""----------PART 1---------"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the Dataset
dataset = pd.read_csv('creditcard.csv')
dataset.dtypes

"""----------PART 2----------"""
# Exploring the Dataset
dataset.columns
dataset.describe()
dataset.Class.value_counts()
#Total number of Frauds and Valids
fraud = dataset[dataset['Class'] == 1]
valid = dataset[dataset['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))
print('Fraudulant Transactions: {}'.format(len(fraud)))
print('Valid Transactions: {}'.format(len(valid)))

# Plotting the Histographs for all the Columns
dataset.hist()
# Exploring the Dataset correlations 
sns.heatmap(dataset.corr(), vmax = 0.5, square = True)

"""----------PART 3----------"""
# Splitting the Dataset into Training and Test Sets
x = dataset.iloc[:, 0:30].values
y = dataset.iloc[:, 30].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

"""----------PART 4----------"""
# Fitting the XGBoost model to the Training Set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

"""
# Fitting the Random Forest CLassification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
"""

# Predicting the Test Set Results
y_pred = classifier.predict(x_test)

"""----------PART 5----------"""
# Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Making the Classification Report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)
# Total Errors in the Predictions
errors = (y_pred != y_test).sum()
print('Total Errors in Prediction = ', errors)

# Applying the K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()