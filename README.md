# QuickPredict
A handy tool that automates some of the tiring parts of ML!

QuickPredict is a GUI that is designed to make Machine Learning more accessible to everyone! Knowledge of ML, or even Python is not necessary to use this tool.


## How to use
- Download all the necessary Python codes onto your machine (This tool requires you to have Python3.x installed and other libraries like tkinter.) Please [click here](https://packaging.python.org/tutorials/installing-packages/) for instructions on how to downlaod the dependencies.
- Open your terminal (or your command prompt) in the folder that contains all the python codes.
- Type: `python3 driver.py` to start the tool

![Your tool should be ready!](https://user-images.githubusercontent.com/46783458/83266724-d8c8b580-a1e0-11ea-8827-58033c3a8153.png)

- Click on **Open Data File** to open your .csv file containing the data-set
- Choose your pre-processing needs and when you're ready, choose the classifier of your choice
- If you choose **Yes** for best parameters, the model uses [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to find the best hyper-paramter for your model.
- Finally, click on **Show Results** to see your output. Voila!

### Note:
This tool is in its _very_ early stages of production. There are quite a few known issues which will be addressed in further releases. New functionality and features would also be added soon.
