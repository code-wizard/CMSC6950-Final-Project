import argparse

import numpy as np
import warnings
from scipy.io import loadmat
from LFSpy import LocalFeatureSelection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from pandas.plotting import table
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from functions import *
np.random.seed(905)

from joblib import dump, load

if __name__ == '__main__':
    df = pd.read_csv('data/iris.csv', sep=',')

    # Accept command line option
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="trained")

    # Parse commandline option
    args, _ = parser.parse_known_args()
    mode = args.mode

    #Read data
    t = pd.read_csv('data/iris.csv', sep=',')
    # Select only two classes out of the three
    clean_df = df.query('species =="setosa" or species=="versicolor"')

    # Describe database and plot table
    desc = clean_df.describe()
    plt.figure(figsize=(10, 2))
    plot = plt.subplot(111, frame_on=False)
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)

    table(plot, desc, loc='best', colColours=["orange", "orange", "orange", "orange"])
    # save the plot as a png file
    plt.title("Dataset Descripion")
    plt.savefig('desc_plot.png')

    # Visualize the data
    plt.figure()
    sns.countplot(clean_df['species'])
    species = clean_df['species'].value_counts()
    plt.ylim(0, 100)
    plt.title('Species Distribution', fontsize=15)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Count (Entries)', fontsize=12)
    plt.grid(color='#ddd', linestyle='-', linewidth=2, alpha=0.2)
    # plt.text(x=-.175, y=100, s='10,422', fontsize=15)
    # plt.text(x=.875, y=100, s='1988', fontsize=15)
    xval = 0
    for index, value in species.items():
        plt.text(x=xval, y=value + 5, s=str(value))
        xval += 1.02
    plt.savefig("classes.png")

    # Extract features and label
    X = clean_df.drop(['species'], axis=1).to_numpy()
    y = clean_df['species'].to_numpy()

    # Label encode the label, convert the clases to 0s and 1s
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)


    def load_dataset(m=0):
        '''
        split dataset
        '''
        features = X
        if m > 0:
            features = add_noise_vars(X, m)
        X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=777, test_size=.2)
        return X_train, X_test, y_train, y_test


    x_train, x_test, y_train, y_test = load_dataset()
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['seagreen', 'darkorange'])
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    # Plot the training points
    s = ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k', alpha=0.6, )
    # Plot the testing points
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend(handles=s.legend_elements()[0], labels=["Setosa", "Versicolor"])
    plt.title("Training data specie distribution by Sepal length and width")
    # plt.show()
    plt.savefig("training.png")
    plt.figure()
    # plt.figure(figsize=(6,5))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend(handles=s.legend_elements()[0], labels=["Setosa", "Versicolor"])
    plt.title("Test data specie distribution by Sepal length and width")
    plt.savefig("test.png")

    if mode == "new":
        score_lfs, y_pred_lfs = results_lfspy(x_train, y_train, x_test, y_test)
        score_des, y_pred_des = results_decision(x_train, y_train, x_test, y_test)
        score_nei, y_pred_nei = results_neighbor(x_train, y_train, x_test, y_test)
        score_bye, y_pred_bye = results_naive(x_train, y_train, x_test, y_test)
        score_list = [score_lfs, score_des, score_nei, score_bye]
        np.save("classification.npy", score_list)
    else:
        score_list = np.load("classification.npy")

    plotScores(score_list, 'Classification Score', 'classification_score.png')

    mlist = np.arange(0, 250, 25)

    if mode == "new":
        # %% Compare across number of noise variables on Iris dataset
        Score_LFS = []
        Score_GB = []
        Score_KNN = []
        Score_TREE = []
        for m in mlist:
            s1, _ = results_lfspy(x_train, y_train, x_test, y_test)
            s2, _ = results_decision(x_train, y_train, x_test, y_test)
            s3, _ = results_neighbor(x_train, y_train, x_test, y_test)
            s4, _ = results_naive(x_train, y_train, x_test, y_test)

            Score_LFS.append(s1)
            Score_TREE.append(s2)
            Score_KNN.append(s3)
            Score_GB.append(s4)
        np.save('result-2-Score_TREE.npy', Score_TREE)
        np.save('result-2-Score_LFS.npy', Score_LFS)
        np.save('result-2-Score_KNN.npy', Score_KNN)
        np.save('result-2-Score_GB.npy', Score_KNN)
    else:
        Score_TREE = np.load('result-2-Score_TREE.npy')
        Score_LFS = np.load('result-2-Score_LFS.npy')
        Score_KNN = np.load('result-2-Score_KNN.npy')
        Score_GB = np.load('result-2-Score_GB.npy')
        #
        # with open('result-2.npy', 'rb') as f:
        #     Score_RFC = np.load(f)
        #     Score_LFS = np.load(f)
        #     Score_SVM = np.load(f)
    # Plot the results
    plt.figure()
    plt.plot(mlist, Score_LFS)
    plt.plot(mlist, Score_TREE)
    plt.plot(mlist, Score_KNN)
    plt.plot(mlist, Score_GB)
    plt.vlines(100, 0, 1.2, linestyles='dashed')
    plt.ylim([0, 1.2])

    # plt.xlabel('Number of Noise Features')
    plt.title('Classification Accuracy by Number of Added Noise Variables', fontsize=14)
    plt.legend(['LFS', 'Desicion Tree', 'KNN', 'Gaussian Bayes'], loc='lower right')
    plt.savefig('classification_noise.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

