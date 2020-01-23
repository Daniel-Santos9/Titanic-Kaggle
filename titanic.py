# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:48:31 2020

@author: Daniel
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer


def convert_sex(valor):
    
    if(valor == 'female'):
        return 1
    else:
        return 0

def convert_embarked(valor):
    if(valor == 'C'):
        return 0
    else:
        if(valor == 'S'):
            return 1
        else:
            return 2


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


classificador = SVC(kernel = 'linear', random_state = 1, C=2)
classificador3 = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
classificador1 = LogisticRegression()

train['Sex_bin'] = train['Sex'].map(convert_sex)
train['Emb_mod'] = train['Embarked'].map(convert_embarked)
train['Age'].fillna(train['Age'].mean(), inplace=True)

Var = ['Sex_bin','Age','Pclass','SibSp','Parch','Fare','Emb_mod']

x = train[Var]
y = train['Survived']

x = x.fillna(-1)

classificador.fit(x,y)

test['Sex_bin'] = test['Sex'].map(convert_sex)
test['Emb_mod'] = test['Embarked'].map(convert_embarked)
test['Age'].fillna(test['Age'].mean(), inplace=True)

x_prev = test[Var]
x_prev = x_prev.fillna(-1)

predict = classificador.predict(x_prev)

sub = pd.Series(predict, index=test['PassengerId'], name='Survived')
sub.shape

sub.to_csv("modelo_SVM.csv", header=True)






