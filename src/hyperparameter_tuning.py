from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Create a hyperparameter grid for Logistic Regression
def random_search_log_reg(X_train,X_test, y_train, y_test):
    log_reg_grid = {
        "C": np.logspace(-4, 4, 20),
        "solver": ["saga"],
        "l1_ratio": np.linspace(0, 1, 10),
    }
    #Tune Logistic Regression

    np.random.seed(42)

    #Setup random hyperparameter search for LogisticRegression
    rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20)

    #Fit random hyperparameter search for LogisticRegression
    rs_log_reg.fit(X_train, y_train)
    rs_log_reg.best_params_
    return rs_log_reg,rs_log_reg.score(X_test, y_test)

#Create a hyperparameter grid for RandomForestClassifier


def random_search_random_forest(X_train, X_test, y_train, y_test):
    rf_grid={"n_estimators": np.arange(10,1000,50),
            "max_depth": [None,10,20],
            "min_samples_split": [2,5],
            "min_samples_leaf": [1,2],
            "bootstrap":[True, False]}
    #Tune RandomForestClassifier

    np.random.seed(42)

    #Setup random hyperparameter search for RandomForest
    rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                            param_distributions=rf_grid,
                            cv=5,
                            n_iter=20)
    #Fit RandomizedSearchCV for RandomForest

    rs_rf.fit(X_train, y_train)
    #Find the best hyperparameters
    rs_rf.best_params_
    # Evaluate the randomized search RandomForestClassifier model
    return rs_rf,rs_rf.score(X_test,y_test)

def grid_search_log_reg(X_train, X_test, y_train, y_test):
    #Different hyperparameters for our LogisticRegression model
    log_reg_grid={"C":np.logspace(-4,4,30),
                "solver":["liblinear"]}

    #Setup grid hyperparameter search for LogisticRegression
    gs_log_reg= GridSearchCV(LogisticRegression(),
                            param_grid=log_reg_grid,
                            cv=5)
    #Fit the model
    gs_log_reg.fit(X_train, y_train)
    #Check the best hyperparams
    gs_log_reg.best_params_
    #Evaluate the model
    return gs_log_reg, gs_log_reg.score(X_test, y_test)

def grid_search_random_forest(X_train, X_test, y_train, y_test):
    # Different hyperparameters for our RandomForest Classifier
    random_forest_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    gs_random_forest= GridSearchCV(RandomForestClassifier(),
                                param_grid=random_forest_grid,
                                cv=5)
    # Fit the model
    gs_random_forest.fit(X_train, y_train)
    gs_random_forest.best_params_
    return gs_random_forest,gs_random_forest.score(X_test, y_test)