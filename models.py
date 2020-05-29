from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from hyper_param_tuning import hyper_param_tuning
import pdb
from sklearn.model_selection import train_test_split

class Models:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_logistic_regression(self, custom_params=False, smart_select=True, param_range=None):
        '''

        :param custom_params: (Bool) True if user wants to use custom params
        :param smart_select: (Bool) True if user wants best params based on a Grid/RandomSearchCV
        :param param_range: (List) Lower bound, upper bound, and step size for searching for params
        :return:
        classfier object and predictions
        '''

        best_params = {'C':1.0} #Default value in sklearn
        if not custom_params and smart_select:
            if param_range is None:
                param_range = {'C':[i for i in np.arange(0.01, 100, 0.1)]}
            best_params = hyper_param_tuning(param_range, self.X_train, self.y_train, classifier=LogisticRegression())
        if custom_params:
            #Take input
            pass
        if not custom_params and not smart_select:
            pass

        clf = LogisticRegression(C=best_params['C'])
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return clf, y_pred


    def do_svm(self, smart_select=True, custom_params=False, param_range=None):
        '''

        :param custom_params: (Bool) True if user wants to use custom params
        :param smart_select: (Bool) True if user wants best params based on a Grid/RandomSearchCV
        :param param_range: (List) Lower bound, upper bound, and step size for searching for params
        :return:
        classfier object and predictions
        '''

        best_params = {'C':1.0} #Default value in sklearn
        if not custom_params and smart_select:
            if param_range is None:
                param_range = {'C':[i for i in np.arange(0.01, 100, 0.1)]}
            best_params = hyper_param_tuning(param_range, self.X_train, self.y_train, classifier=SVC())
        if custom_params:
            #Take input
            pass
        if not custom_params and not smart_select:
            pass

        clf = SVC(C=best_params['C'])
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return clf, y_pred

    def do_knn(self, smart_select=True, custom_params=False, param_range=None):
        '''

        :param custom_params: (Bool) True if user wants to use custom params
        :param smart_select: (Bool) True if user wants best params based on a Grid/RandomSearchCV
        :param param_range: (List) Lower bound, upper bound, and step size for searching for params
        :return:
        classfier object and predictions
        '''

        best_params = {'n_neighbors':5} #Default value in sklearn
        if not custom_params and smart_select:
            if param_range is None:
                param_range = {'n_neighbors':[i for i in range(1, len(self.X_test), 1)]}
            best_params = hyper_param_tuning(param_range, self.X_train, self.y_train, classifier=KNeighborsClassifier())
        if custom_params:
            #Take input
            pass
        if not custom_params and not smart_select:
            pass

        clf = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return clf, y_pred
