import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from backend import *
from ttkthemes import themed_tk
from models import Models
from tkinter import filedialog
from data_preprocessing import *


def main():
    root = themed_tk.ThemedTk()
    root.get_themes()
    root.set_theme('radiance')

    #Heading
    def display_header():
        header = Label(root, text='QuickPredict', font=('Arial Bold', 20))
        header.grid(column=0, row=0, pady=10)

    def button_open_file():
        #Button to open file
        bt_openfile = Button(root, text='Open Data File', command=open_file)
        bt_openfile.grid(column=0, row=1, pady=10) #Set command=open_file

    def options_preprocessing():
        # Sub-heading: What all do you want?
        header = Label(root, text='Please check all the pre-processing that you\'d like to do')
        header.grid(column=0, row=2, pady=10)

    def checkboxes_preprocessing():
        preproc = Preprocess
        var_normalize_f, var_standardize_f, var_labelencode_r = tk.IntVar(), tk.IntVar(), tk.IntVar()
        standard_features = Checkbutton(root, text='Standardize features i.e, StandardScaler()', variable=var_standardize_f, onvalue=1, offvalue=0, command=do_standardize_features)
        standard_features.grid(column=0, row=3)
        normalize_features = Checkbutton(root, text='Normalize features i.e, Normalizer()', variable=var_normalize_f, onvalue=1, offvalue=0, command=do_normalize_features)
        normalize_features.grid(column=0, row=4)
        label_encode_response = Checkbutton(root, text='Label Encode response', variable=var_labelencode_r, onvalue=1, offvalue=0, command=do_encode_response)
        label_encode_response.grid(column=0, row=5)
    def display_subheading_chooseclassifier():
        #Sub-heading: Choose your classifier
        header = Label(root, text='Choose your classifier')
        header.grid(column=0, row=7, pady=10)


    def choose_classifier():
        '''
        Code taken from: https://pythonspot.com/tk-dropdown-example/
        Pop up menu to choose classifier
        :return:
        returns the chosen classifier
        '''
        mainframe = Frame(root)
        mainframe.grid(column=0, row=8, sticky=(N, W, E, S))
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)
        #mainframe.pack(pady=100, padx=100)
        ##
        # Create a Tkinter variable
        tkvar = StringVar(root)

        # Dictionary with options
        choices = {'Support Vector Machine (SVM)', 'Logistic Regression', 'K-Nearest Neighbours (KNN)'}
        tkvar.set('Choose..')  # set the default option

        popupMenu = OptionMenu(mainframe, tkvar, *choices)
        popupMenu.grid(row=9, column=0)

        # on change dropdown value
        def change_dropdown(*args):
            perform_classification(tkvar.get())

        #myButton = Button(root, text='Result for -> ' + tkvar.get(), command=display_result)
        #myButton.pack()

        # link function to change dropdown
        tkvar.trace('w', change_dropdown)


    def button_hyperparamters():
        '''
        Code taken from: https://www.python-course.eu/tkinter_radiobuttons.php
        '''
        #root = tk.Tk()

        v = tk.IntVar()
        v.set(1)  # initializing the choice, i.e. Python

        choices = [
            ("No"), #No = 0
            ("Yes") #Yes = 1
        ]

        def ShowChoice():
            set_hyperparam_choice(v.get())
            print(v.get())


        tk.Label(root,
                 text="""Do you want the model to select the best parameters? (Takes more time to execute)""",
                 justify=tk.LEFT,
                 padx=20).grid(pady=5)

        for val, choice in enumerate(choices):
            tk.Radiobutton(root,
                           text=choice,
                           padx=20,
                           variable=v,
                           command=ShowChoice,
                           value=val).grid(pady=10)


    def button_result():
        #Get result
        bt_result = Button(root, text='Show results', command=show_result)
        bt_result.grid(column=0, row=15) #Set command=get_result

    def show_result():
        accuracy, conf_matrix, precision, recall, f1_score = display_result()
        ########
        text_box = tk.Text(root, height=10, width=100)
        text_box.grid(column=0, row=20)
        result='''
Accuracy        = {0}
Precision       = {1}
Recall          = {2}
F1-Score        = {3}
Confusion Matrix: 
{4}
        '''.format(accuracy, precision, recall, f1_score, conf_matrix)
        text_box.insert(tk.END, result)




    display_header()
    button_open_file()
    options_preprocessing()
    checkboxes_preprocessing()
    display_subheading_chooseclassifier()
    choose_classifier()
    button_hyperparamters()
    button_result()


    root.mainloop()
