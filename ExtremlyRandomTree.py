# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:37:56 2019

@author: HuuTon
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import glob
import os
import numpy as np
import math
import random
X=[]
y=[]
index=0
for name in glob.glob("/home/tonlh/Desktop/Code/Imorph/Data/LeftWingsFeaturesIndex9/*_nev_sSub.txt"):
    print (name)
    with open(name) as f:
        lines = f.readlines()
        for line in lines:
            numbers = list(map(float, line.split()))
            X.append(numbers)
            y.append(0)
    index=index+1
    if (index==10):
        break
    # print(len(X))
index=0
for name in glob.glob("/home/tonlh/Desktop/Code/Imorph/Data/LeftWingsFeaturesIndex9/*_pos_sSub.txt"):
    print (name)
    with open(name) as f:
        lines = f.readlines()
        for line in lines:
            numbers = list(map(float, line.split()))
            X.append(numbers)
            y.append(1)
    index=index+1
    if (index==10):
        break
    # print(len(X))
print (len(X))
print (len(y))
X, y = shuffle(X, y)
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print (scores)

        