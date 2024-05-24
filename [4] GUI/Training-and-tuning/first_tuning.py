"""
This document is for testing which models we are going to use. Exploration is being done and this document is not structured in a way that is suitable for production.
"""

# ------ import ------ #
import numpy as np
import pandas as pd
from util import computeFeatureImportance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV as GSCV

np.random.seed(42)

# ------ Settings ------ #

# gridsearch
do_gridsearch_svc = False
do_gridsearch_knn = False
do_gridsearch_rf = False

# randomsearch
do_randomsearch_svc = False
do_randomsearch_rf = False

# feature importance
do_feature_importance = True
# ------ Data import ------ #

print("Importing data...")
x = pd.read_csv(r'Data Gathering and Preprocessing/features_Walking_scaled.csv')
print("Data imported")

# ------ train, test split ------ #

print("shuffling data and splitting data into train and test...")
train, test = train_test_split(x, train_size=0.8, shuffle=True)

# ------ x, y split ------ #

print("Splitting data into x and y...")
le = LabelEncoder()
le.fit(train["label"])
print(f"Classes: {le.classes_}")

y_train = le.transform(train["label"])
x_train = train.copy()
x_train = x_train.drop(["label", "time", "ID"], axis=1)

y_test = le.transform(test["label"])
x_test = test.copy()
x_test = x_test.drop(["label", "time", "ID"], axis=1)


# ------ gridsearch svc ------ #
    
if do_gridsearch_svc:
    parameters = {'kernel':('linear', 'rbf'), 
                  'C':[0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
    model = SVC()
    clf = GSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for gridsearch svc:")
    print(clf.best_params_)
    print(clf.best_score_)

# ------ gridsearch knn ------ #

if do_gridsearch_knn:
    parameters = {'n_neighbors': range(1, 101), 'weights':('uniform', 'distance')}
    model = KNN()
    clf = GSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for gridsearch knn:")
    print(clf.best_params_)
    print(clf.best_score_)
    
# ------ gridsearch rf ------ #

if do_gridsearch_rf:
    parameters = {'n_estimators': range(1, 101), 'max_depth': range(1, 21), 'criterion':('gini', 'entropy')}
    model = RF()
    clf = GSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for gridsearch rf:")
    print(clf.best_params_)
    print(clf.best_score_)

# ------ feature importance ------ #

if do_feature_importance:
    print("calculating feature importance...")
    imp = computeFeatureImportance(x_train, y_train, n_repeats=50)
    total = imp["feature_importance"].sum()
    imp["feature_importance"] = imp["feature_importance"] / total
    print(imp)