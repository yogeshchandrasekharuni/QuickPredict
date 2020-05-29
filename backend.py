import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from models import Models
from performance_metrics import Metrics
from data_preprocessing import Preprocess
import pdb
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils
import sklearn.utils._cython_blas

data = None
X_train = X_test = y_train = y_test = None
HYPERPARAM_RB_CHOICE = 1
final_predictions = None
#prepoc = Preprocess(X_train, X_test, y_train, y_test)

def open_file():
    '''
    Opens a CSV file
    :return:
    pandas dataframe containing the csv file
    '''
    file = askopenfilename(title= 'Select File', filetypes = (('CSV Files', '*.csv'), ))
    if file is not None:
        data = pd.read_csv(file)
    '''
    else:
        #Show error
        messagebox.showinfo('Error','Unsupported file format!')
    '''
    #Getting the data too
    X, y = get_data(data)
    #X, y = preprocess_data(X, y)
    split_data(X, y)
    global preproc
    #pdb.set_trace()
    preproc = Preprocess(X_train, X_test, y_train, y_test)

def perform_classification(classifier):
    model = Models(X_train, X_test, y_train, y_test)
    print('Peforming ' + classifier)
    if classifier == 'Logistic Regression':
        clf, predictions = model.do_logistic_regression(smart_select=get_hyperparam_choice(), custom_params=False,
                                                        param_range=None)
    if classifier == 'Support Vector Machine (SVM)':
        clf, predictions = model.do_svm(smart_select=get_hyperparam_choice(), custom_params=False, param_range=None)
    if classifier == 'K-Nearest Neighbours (KNN)':
        clf, predictions = model.do_knn(smart_select=get_hyperparam_choice(), custom_params=False, param_range=None)

    global final_predictions
    final_predictions = predictions
    score = accuracy_score(y_test, predictions)
    print('Accuracy = ', score)

def display_result():
    if final_predictions is not None:
        metric = Metrics(y_test, final_predictions)
        confusion_matrix = metric.get_confusion_matrix()
        precision, recall, f1_score, _ = metric.get_prec_recall_f1()
        print(_)
        accuracy = str(metric.get_accuracy())
        return accuracy, confusion_matrix, precision, recall, f1_score
    #label = Label(root, text='Accuracy = ' + str(metric.get_accuracy()))
    #label.pack()


def set_hyperparam_choice(current_choice):
    global HYPERPARAM_RB_CHOICE
    HYPERPARAM_RB_CHOICE = current_choice

def get_hyperparam_choice():
    return HYPERPARAM_RB_CHOICE


def get_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
def preprocess_data(X, y):
    #Encoding the response variable
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    #Scaling X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y
def split_data(X, y):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y)


def do_standardize_features():
    global X_train, X_test
    X_train, X_test = preproc.standardize_features()
    print(preproc.X_train)

def do_normalize_features():
    global X_train, X_test
    X_train, X_test = preproc.normalize_features()
    print(preproc.X_train)

def do_encode_response():
    global y_train, y_test
    y_train, y_test = preproc.labelencode_responses()
    print(preproc.y_train)


