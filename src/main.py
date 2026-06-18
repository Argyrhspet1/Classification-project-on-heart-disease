#Import the tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
from load_data import *
from modelling import *
from hyperparameter_tuning import *
from evaluation import *
from feature_importance import *
from sklearn.metrics import confusion_matrix


data=load_data() #load the heart disease data

X_train, X_test, y_train, y_test, X=train_test(data) #train-test-split the data into subsets for training and testing

fit_and_score(X_train, X_test, y_train, y_test) #fit the models in the data and calculate their score

rs_log_reg, rs_log_acc=random_search_log_reg(X_train, X_test, y_train, y_test) # tune hyperparameters using sklearns random search cross validation and find the most appropriate value hyperparameters for logistic regression model among 20 combinations
print(f"Randomized Search Cross Validated Logistic Regression achieved : {rs_log_acc} accuracy.")

gs_log_reg, gs_log_acc=grid_search_log_reg(X_train, X_test, y_train, y_test) # tune hyperparameters using sklearns grid search cross validation and find the most appropriate value hyperparameters for logistic regression model among all available combinations
print(f"Grid Search Cross Validated Logistic Regression achieved : {gs_log_acc} accuracy.")

rs_random_forest,rs_random_forest_acc=random_search_random_forest(X_train, X_test, y_train, y_test) # tune hyperparameters using sklearns random search cross validation and find the most appropriate value hyperparameters for Random Forest Classifier model among 20 combinations
print(f"Randomized Search Cross Validated Random Forest achieved : {rs_random_forest_acc} accuracy.")

gs_random_forest, gs_random_forest_acc=grid_search_random_forest(X_train, X_test, y_train, y_test) # tune hyperparameters using sklearns grid search cross validation and find the most appropriate value hyperparameters for Random Forest Classifier model among all available combinations
print(f"Grid Search Cross Validated Random Forest achieved : {gs_random_forest_acc} accuracy.")

roc_curve_display(X_test, y_test, gs_log_reg, gs_random_forest) #Evaluation and comparison between RF Classifier and Logistic Regression using RO Curve and Area Under the curve metrics

y_pred= predict(X_test, gs_log_reg) #predict using tuned grid search cv logistic regression
y_pred2= predict(X_test, gs_random_forest) #predict using tuned grid search cv random forest classifier

confusion_mat(y_test, y_pred, y_pred2) #confusion matrix that depicts and compares logistic regression and random forest scores

coeffs=feature_importance_log_reg(X_test, y_test) #computes the effect of each logistic regression coefficient in the results

visualize_features_log_reg(X,coeffs) #visualize coefficients effect

coeffs_random_forest= feature_importance_random_forest(X_test, y_test) #computes the effect of each Random Forest Classifier coefficient in the results
print(coeffs_random_forest)

visualize_features_random_forest(X,coeffs_random_forest)  #visualize coefficients effect