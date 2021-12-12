import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost
from sklearn.metrics import accuracy_score

code_path = os.getcwd()
data_path=r'C:\Users\mbiswas\OneDrive - Capgemini\Documents'
os.chdir(data_path)
fruits = pd.read_table('fruit_data_with_colors.txt')
os.chdir(code_path)

X = fruits['feature_names']
y = fruits['fruit_label']

LabelEncoder = preprocessing.LabelEncoder().fit(y)
y1 = LabelEncoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logistic_score = logreg.score(X_test, y_test)

clf = DecisionTreeClassifier().fit(X_train, y_train)
Decisiontree_score = clf.score(X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(X, y1, random_state=0)
xgb_model = xgboost.XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)
y_pred = xgb_model.fit(X_train, y_train).predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
xgb_score = xgb_model.score(X_test, y_test)