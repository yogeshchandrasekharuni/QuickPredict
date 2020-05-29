from sklearn.model_selection import RandomizedSearchCV


def hyper_param_tuning(param_range, X_train, y_train, classifier):
    print('Selecting best paramters now...')
    best_params = RandomizedSearchCV(estimator=classifier, param_distributions=param_range)
    best_params.fit(X_train, y_train)
    return best_params.best_params_