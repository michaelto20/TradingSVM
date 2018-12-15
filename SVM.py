# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:55:31 2018

@author: Michael Townsend
"""

import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize



def main():
    inputData = 'C:\\Users\\Michael Townsend\\Desktop\\StockInfo\\FOREX\\Data\\BollingerBands.csv'
    data = pd.read_csv(inputData)    
    
    # struggling to normalize datetime so for now just drop that column
    data = data.drop(data.columns[0], axis = 1)
    data = data.drop(['DateTimeStamp'], axis = 1)
    data = data.dropna()
    
    # Separate out the x_data and y_data.
    x_data = data.loc[:, data.columns != "SellWins"]
    y_data = data.loc[:, "SellWins"]
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state = None, shuffle=True)
    
    # normalize data for SVM
    norm_Xtrain = normalize(x_train)
    norm_Xtest = normalize(x_test)
    
    clf = SVC(C=0.001, cache_size = 200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='linear', max_iter = -1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False).fit(norm_Xtrain, y_train)
    
    predicted_test = clf.predict(norm_Xtest)
    predicted_train = clf.predict(norm_Xtrain)
    
    print("test: "+ str(accuracy_score(y_test, predicted_test)))
    print("train: "+ str(accuracy_score(y_train, predicted_train)))
    
    param_gridCLF = { 
    'C': [0.001, 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
    }
    
    gridcvCLF = GridSearchCV(estimator=clf, param_grid=param_gridCLF, cv= 10)
    gridcvCLF.fit(norm_Xtrain, y_train)
    print(gridcvCLF.best_params_)
    print(gridcvCLF.best_score_)
    predictedTuneCLF = gridcvCLF.predict(norm_Xtest)
    print("tuned CLF accuracy: "+ str(accuracy_score(y_test, predictedTuneCLF)))
    
    
    
    
main()