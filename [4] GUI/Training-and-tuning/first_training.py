"""
This document is for testing which models we are going to use. Exploration is being done and this document is not structured in a way that is suitable for production.
"""
# ------ import ------ #
import time
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

n_clusters = 4
train_size = 0.8

# supervised learning
do_knn = True
do_svc = True

# random forest
do_rf = True
plot_rf = True
plot_rf_tree = True

# other models
do_dt = True
do_bagged_class = True
do_gd = True
do_gnb = True

# clustering
do_kmeans = True
do_kmeans_plot = True

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
x = pd.read_csv(r'Preprocessed-data\Walking\features_Walking_scaled.csv')
print("Data imported")
print(f"Shape of data: {x.shape}")

# ------ train, test split ------ #

print("shuffling data and splitting data into train and test...")
train, test = train_test_split(x, train_size=train_size, shuffle=True)

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

if do_kmeans:
    print("starting kmeans...")
    # ------ PCA ------ #

    print("Using PCA...")
    pca = PCA(2)
    df = pca.fit_transform(x_train)
    df_test = pca.fit_transform(x_test)
    
    x_train_pca = np.array(df)
    x_test_pca = np.array(df_test)

    # ------ Training KMeans ------ #

    print("Training KMeans...")
    start_time = time.perf_counter()
    model = KMeans(n_clusters=n_clusters, n_init=10)
    model.fit(x_train)
    label = model.labels_
    # print(label)

    pred_y = model.predict(x_test)
    print(f"KMeans accuracy: {accuracy_score(y_test, pred_y)}")
    print(f"KMeans time: {time.perf_counter() - start_time:.2f}")	

    model = KMeans(n_clusters=n_clusters)
    model.fit(df)
    label = model.labels_

    # ------ centroid ------ #

    print("Calculating centroids...")
    centroids = model.cluster_centers_
    u_labels = np.unique(label)
    
    cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'brown', 5: 'purple', 6: 'orange', 7: 'pink'}
    ldict = {}
    for i in range(len(u_labels)):
        ldict[i] = le.classes_[i]

    pred = model.predict(df_test)

    if do_kmeans_plot:
        # ------ plot ------ #
        # for i, data in enumerate(x):
        #     print(f"{data[0]=}, {data[1]=}, {label[i]=}")
        #     plt.scatter(data[0], data[1], label=label[i])

        print("Plotting model and test data...")
        
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].title.set_text('Model')
        for l in u_labels:
            ii = np.where(label == l)
            axs[0, 0].scatter(x_train_pca[ii, 0], x_train_pca[ii, 1], c=cdict[l], label=ldict[l])
        # axs[0, 0].scatter(df[:, 0], df[:, 1], c=label)

        axs[0, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        axs[0, 0].legend()
        
        # ------ prediction test data ------ #
        # Make predictions on the test data

        axs[0, 1].title.set_text('new data points')
        axs[0, 1].scatter(df_test[:, :1], df_test[:, 1:], c='grey')
        axs[0, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')

        # create second plot which show new points whichout prediction
        axs[1, 0].title.set_text('New data on model')
        for l in u_labels:
            ii = np.where(label == l)
            axs[1, 0].scatter(x_train_pca[ii, 0], x_train_pca[ii, 1], c=cdict[l], label=ldict[l])
        axs[1, 0].scatter(df_test[:, :1], df_test[:, 1:], c='grey')
        axs[1, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        axs[1, 0].legend()

        # create third plot which show the predictions of the new points
        axs[1, 1].title.set_text('result')
        for l in u_labels:
            ii = np.where(label == l)
            axs[1, 1].scatter(x_train_pca[ii, 0], x_train_pca[ii, 1], c=cdict[l], label=ldict[l])
        for l in u_labels:
            ii = np.where(pred == l)
            axs[1, 1].scatter(x_test_pca[ii, 0], x_test_pca[ii, 1], c=cdict[l])
        axs[1, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        plt.legend()
        plt.show()

        print("plotting model vs actual...")
        fig, axs = plt.subplots(2)
        axs[0].title.set_text('model result')
        for l in u_labels:
            ii = np.where(label == l)
            axs[0].scatter(x_train_pca[ii, 0], x_train_pca[ii, 1], c=cdict[l], label=ldict[l])
        axs[0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        axs[0].legend()
        
        axs[1].title.set_text('Actual result')
        for l in u_labels:
            ii = np.where(y_train == l)
            axs[1].scatter(x_train_pca[ii, 0], x_train_pca[ii, 1], c=cdict[l], label=ldict[l])
        axs[1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
        axs[1].legend()
        plt.show()

# ------ knn ------ #

if do_knn:
    print("Training KNN model...")
    start_time = time.perf_counter()
    knn = KNN(n_neighbors=3)
    knn.fit(x_train, y_train)

    y_pred_train = knn.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = knn.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"knn accuracy: {accuracy_train=}, {accuracy_test=}")
    print(f"knn time: {time.perf_counter() - start_time:.2f}s")

# ------ svc ------ #

if do_svc:
    print("Training SVC model...")
    svc = SVC()
    svc.fit(x_train, y_train)

    y_pred_train = svc.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = svc.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"svc accuracy: {accuracy_train=}, {accuracy_test=}")
    print(f"svc time: {time.perf_counter() - start_time:.2f}")
    
# ------ random forest ------ #

if do_rf:
    print("Training Random Forest model...")
    start_time = time.perf_counter()
    rf = RF()
    rf.fit(x_train, y_train)

    y_pred_train = rf.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = rf.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"rf accuracy: {accuracy_train=}, {accuracy_test=}")
    print(f"rf score: {rf.score(x_test, y_test)=}")
    print(f"rf time: {time.perf_counter() - start_time:.2f}")
    if plot_rf:
        if not do_kmeans:
            raise Exception("Kmeans (do_kmeans = True) must be enabled to plot RF model vs actual")
        print("Plotting RF model vs actual...")
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
        print("Plotting individual RF tree...")
        estimator = rf.estimators_[5]
        
        fig = plt.figure(figsize=(15, 10))
        plot_tree(estimator, filled=True, feature_names=x.columns, class_names=le.classes_, impurity=True, rounded=True)
        plt.show()

# ------ decision tree ------ #

if do_dt:
    print("training decision tree...")
    start_time = time.perf_counter()
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)

    y_pred_train = dt.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = dt.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    fig = plt.figure(figsize=(25,20))
    plot_tree(dt, filled=True)
    plt.show()


    print(f"dt accuracy: {accuracy_train=}, {accuracy_test=}")
    print(f"dt time: {time.perf_counter() - start_time:.2f}")

# ------ bagged classifier ------ #

if do_bagged_class:
    print("training bagged classifier...")
    start_time = time.perf_counter()
    chosen = SVC()
    num_models = 100
    model = BaggingClassifier(estimator=chosen, n_estimators=num_models, random_state=42)
    
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = model.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"bagged accuracy: {str(chosen)}: {accuracy_train=}, {accuracy_test=}")
    print(f"bagged time: {time.perf_counter() - start_time:.2f}")

# ------ gradient descent ------ #

if do_gd:
    print("training SGD classifier...")
    start_time = time.perf_counter()
    gd = SGDClassifier()
    gd.fit(x_train, y_train)

    y_pred_train = gd.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = gd.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"sgd accuracy: {accuracy_train=}, {accuracy_test=}")
    print(f"sgd time: {time.perf_counter() - start_time:.2f}")

# ------ guassian naive Bayes ------ #

if do_gnb:
    print("training gaussian naive bayes...")
    start_time = time.perf_counter()
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    y_pred_train = nb.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = nb.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"gaussian nb accuracy: {accuracy_train=}, {accuracy_test=}")
    print(f"gaussion nb time: {time.perf_counter() - start_time:.2f}")


# ------ gridsearch svc ------ #
    
if do_gridsearch_svc:
    print("training gridsearch svc...")
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
    print("training gridsearch rf...")
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
    print("training randomsearch svc...")
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
    print("training randomsearch rf...")
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
    imp = computeFeatureImportance(x_train, y_train, n_repeats=50, plotting=True)
    # total = imp["feature_importance"].sum()
    # imp["feature_importance"] = imp["feature_importance"] / total
    print(imp)