"""
This document is for testing which models we are going to use. Exploration is being done and this document is not structured in a way that is suitable for production.
"""
# ------ import ------ #
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import RandomizedSearchCV as RSCV
from util import computeFeatureImportance
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)

# ------ Settings ------ #

n_clusters = 3

# supervised learning
do_knn = False
do_svc = False

# random forest
do_rf = True
plot_rf = False
plot_rf_tree = False
plot_trees = True

do_dt = False
do_bagged_class = False
do_gd = False
do_gnb = False

# clustering
do_kmeans_plot = False

# gridsearch
do_gridsearch_svc = False
do_gridsearch_rf = False

# randomsearch
do_randomsearch_svc = False
do_randomsearch_rf = False

# feature importance
do_feature_importance = True

# ------ Data import ------ #
# x = []
# with open('Data Gathering and Preprocessing/features_Walking_scaled.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
    
#     for row in reader:
#         x.append(row)

print("Importing data...")
x = pd.read_csv(r'Data Gathering and Preprocessing/features_Walking_scaled.csv')
print("Data imported")

# ------ train, test split ------ #

print("Splitting data into train and test...")
train, test = train_test_split(x, train_size=0.8)

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

# ------ PCA ------ #

print("Using PCA...")
pca = PCA(2)
df = pca.fit_transform(x_train)
df_test = pca.fit_transform(x_test)
# df = pd.DataFrame(df)
# df_test = pd.DataFrame(df_test)
# print(df)

# ------ Training KMeans ------ #

print("Training KMeans...")
model = KMeans(n_clusters=n_clusters)
model.fit(x_train)
label = model.labels_
# print(label)

pred_y = model.predict(x_test)
print(f"KMeans accuracy: {accuracy_score(y_test, pred_y)}")

model = KMeans(n_clusters=n_clusters)
model.fit(df)
label = model.labels_

# ------ centroid ------ #

print("Calculating centroids...")
centroids = model.cluster_centers_
u_labels = np.unique(label)

pred = model.predict(df_test)

if do_kmeans_plot:
    # ------ plot ------ #
    # for i, data in enumerate(x):
    #     print(f"{data[0]=}, {data[1]=}, {label[i]=}")
    #     plt.scatter(data[0], data[1], label=label[i])

    print("Plotting model and test data...")
    
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].title.set_text('Model')
    axs[0, 0].scatter(df[:, :1], df[:, 1:], c=label)
    axs[0, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        
    # ------ prediction test data ------ #

    # Make predictions on the test data

    axs[0, 1].title.set_text('new data points')
    axs[0, 1].scatter(df_test[:, :1], df_test[:, 1:], c='red')
    axs[0, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')

    # create second plot which show new points whichout prediction
    axs[1, 0].title.set_text('New data on model')
    axs[1, 0].scatter(df[:, :1], df[:, 1:], c=label)
    axs[1, 0].scatter(df_test[:, :1], df_test[:, 1:], c='red')
    axs[1, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')

    # create third plot which show the predictions of the new points
    axs[1, 1].title.set_text('result')
    axs[1, 1].scatter(df[:, :1], df[:, 1:], c=label)
    axs[1, 1].scatter(df_test[:, :1], df_test[:, 1:], c=pred)
    axs[1, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
    plt.legend()
    plt.show()

    print("plotting model vs actual...")
    fig, axs = plt.subplots(2)
    axs[0].title.set_text('model result')
    axs[0].scatter(df[:, :1], df[:, 1:], c=label)
    axs[0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
    
    axs[1].title.set_text('Actual result')
    axs[1].scatter(df[:, :1], df[:, 1:], c=y_train)
    axs[1].legend()
    axs[1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
    plt.show()

# ------ knn ------ #

if do_knn:
    knn = KNN(n_neighbors=3)
    knn.fit(x_train, y_train)

    y_pred_train = knn.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = knn.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"knn: {accuracy_train=}, {accuracy_test=}")

# ------ svc ------ #

if do_svc:
    svc = SVC()
    svc.fit(x_train, y_train)

    y_pred_train = svc.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = svc.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"svc: {accuracy_train=}, {accuracy_test=}")
    
# ------ random forest ------ #

if do_rf:
    rf = RF()
    rf.fit(x_train, y_train)

    y_pred_train = rf.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = rf.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"rf: {accuracy_train=}, {accuracy_test=}")
    if plot_rf:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].title.set_text('kmeans model result')
        axs[0, 0].scatter(df[:, :1], df[:, 1:], c=label)
        axs[0, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        
        axs[0, 1].title.set_text('Actual result')
        axs[0, 1].scatter(df[:, :1], df[:, 1:], c=y_train)
        axs[0, 1].legend()
        axs[0, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        
        axs[1, 0].title.set_text('rf model result')
        axs[1, 0].scatter(df[:, :1], df[:, 1:], c=y_pred_train)
        axs[1, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        
        axs[1, 1].title.set_text('Actual result')
        axs[1, 1].scatter(df[:, :1], df[:, 1:], c=y_train)
        axs[1, 1].legend()
        axs[1, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        plt.show()

    if plot_rf_tree:
        estimator = rf.estimators_[5]
        
        fig = plt.figure(figsize=(15, 10))
        plot_tree(estimator, filled=True, feature_names=x.columns, class_names=le.classes_, impurity=True, rounded=True)
        plt.show()

    if plot_trees:
        var_pattern_corr = 'class_corr_{}'
        var_pattern_incorr = 'class_incorr_{}'
        start_val = 0
        corr = []
        incorr = []
        var_pattern_acc = 'acc_class_{}'
        acc = []

        for i in range(0, len(le.classes_)):
            globals()[var_pattern_corr.format(i)] = start_val
            globals()[var_pattern_incorr.format(i)] = start_val
            corr.append(globals()[var_pattern_corr.format(i)])
            incorr.append(globals()[var_pattern_incorr.format(i)])
        
        # Check accuracy per class
        for tree in rf.estimators_:
            test_pred = tree.predict(x_test)
            for i in range(len(y_test)):
                for x in range(0, len(le.classes_)):
                    if test_pred[i] == x and y_test[i] == x:
                        corr[x] += 1
                    elif test_pred[i] == x and y_test[i] != x:
                        incorr[x] += 1
            print(test_pred)

        bins = le.classes_
        y_pos = np.arange(len(bins))    
        plt.bar(y_pos - 0.2, corr, 0.4, label='Correctly labeled')
        plt.bar(y_pos + 0.2, incorr, 0.4, label='Incorrectly labeled')
        plt.xticks(y_pos, bins)
        plt.xlabel("Distribution of classification")
        plt.ylabel("Number of classifications")
        plt.legend()
        plt.show()
        
        for x in range(0, len(corr)):
            if corr[x] != 0 and incorr[x] != 0:
                globals()[var_pattern_acc.format(x)] = corr[x] / (corr[x] + incorr[x])
            else:
                globals()[var_pattern_acc.format(x)] = 0
            acc.append(globals()[var_pattern_acc.format(x)])
        print(corr)    
        print(incorr)
        count = 0
        for item in acc:
            print("The accuracy for class named {} is: ".format(le.classes_[count]) + str(item))
            count += 1
    
# ------ decision tree ------ #

if do_dt:
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)

    y_pred_train = dt.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = dt.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    fig = plt.figure(figsize=(25,20))
    plot_tree(dt, filled=True)


    print(f"dt: {accuracy_train=}, {accuracy_test=}")

# ------ bagged classifier ------ #

if do_bagged_class:
    chosen = SVC()
    num_models = 100
    model = BaggingClassifier(base_estimator=chosen, n_estimators=num_models, random_state=42)
    
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = model.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"bagged {str(chosen)}: {accuracy_train=}, {accuracy_test=}")

# ------ gradient descent ------ #

if do_gd:
    gd = SGDClassifier()
    gd.fit(x_train, y_train)

    y_pred_train = gd.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = gd.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"sgd: {accuracy_train=}, {accuracy_test=}")

# ------ guassian naive Bayes ------ #

if do_gnb:
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    y_pred_train = nb.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = nb.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"gaussian nb: {accuracy_train=}, {accuracy_test=}")


# ------ gridsearch svc ------ #
    
if do_gridsearch_svc:
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    model = SVC()
    clf = GSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for gridsearch svc:")
    print(clf.best_params_)
    print(clf.best_score_)
    
    
# ------ gridsearch rf ------ #

if do_gridsearch_rf:
    parameters = {'n_estimators':[1, 10, 100, 1000], 'max_depth':[None], 'min_samples_split':[2, 4, 8]}
    model = RF()
    clf = GSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for gridsearch rf:")
    print(clf.best_params_)
    print(clf.best_score_)
    
# ------ randomsearch svc ------ #

if do_randomsearch_svc:
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    model = SVC()
    clf = RSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for randomsearch svc:")
    print(clf.best_params_)
    print(clf.best_score_)
    
# ------ randomsearch rf ------ #    

if do_randomsearch_rf:
    parameters = {'n_estimators':[1, 10, 100, 1000], 'max_depth':[None], 'min_samples_split':[2, 4, 8]}
    model = RF()
    clf = RSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for randomsearch rf:")
    print(clf.best_params_)
    print(clf.best_score_)

# ------ feature importance ------ #
    
if do_feature_importance:
    print("calculating feature importance...")
    imp = computeFeatureImportance(x_train, y_train)
    total = imp["feature_importance"].sum()
    imp["feature_importance"] = imp["feature_importance"] / total
    print(imp)