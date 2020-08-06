
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


def plotScores(scores, title=None, filename="plot.png"):
    '''
    Plot classification scores.
    '''
    plt.figure()
    plt.bar(['Tree', 'GBayes', 'KNN', 'LFS'], scores, color='lightseagreen')
    plt.ylim([0, 1])
    plt.xlabel("Classifier")
    plt.ylabel("Prediction")
    plt.xticks(color='darkslategray')
    plt.yticks(color='tomato')

    for i, v in enumerate(scores):
        plt.text(i - 0.1, 0.4, '{:.{}f}'.format(v, 2), size=12)

    plt.title(title, fontsize=14)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    return None


def add_noise_vars(x, m, std_range=[0, 3]):
    '''
    Adds m Gaussian noise variables to data array x. Gaussian distribution have
    zero mean with standard deviations sampled from a uniform distribution on
    std_range.
    '''
    n = x.shape[0]
    stds = np.random.uniform(low=std_range[0], high=std_range[1], size=m)
    noise_vars = [np.random.normal(loc=0.0, scale=s, size=[n ,]) for s in stds]
    return np.hstack((x, np.stack(noise_vars).T))


def results_lfspy(x_train, y_train, x_test, y_test):
    '''
    Trains an tests and LFS model using default parameters on the given dataset.
    '''
    print('Training and testing an LFS model with default parameters.\nThis may take a few minutes...')
    lfs = LocalFeatureSelection(rr_seed=777)
    pipeline = Pipeline([('classifier', lfs)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)
    return score, y_pred


def results_decision(x_train, y_train, x_test, y_test):
    print('Training and testing a Decision Forest with default parameters.')
    desc = DecisionTreeClassifier(max_depth=5, random_state=777)
    pipeline = Pipeline([('classifier', desc)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)

    return score, y_pred


def results_neighbor(x_train, y_train, x_test, y_test):
    print('Training and testing a K-Neighbors with default parameters')
    k = KNeighborsClassifier(3)
    #     sel = SelectKBest(f_classif, k=int(0.25*x_train.shape[1]))
    pipeline = Pipeline([('classifier', k)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)

    return score, y_pred


def results_naive(x_train, y_train, x_test, y_test):
    print('Training and testing a Gaussian Naive Bayes with default parameters ')
    naive = GaussianNB()
    #     sel = SelectKBest(f_classif, k=int(0.25*x_train.shape[1]))
    pipeline = Pipeline([('classifier', naive)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)

    return score, y_pred