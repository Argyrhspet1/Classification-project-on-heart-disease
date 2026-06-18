from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This function calculates the effect of each logistic Regression coefficient in the score
def feature_importance_log_reg(X_train, y_train):
    clf = LogisticRegression(C=0.01610262027560939, solver= 'liblinear')
    clf.fit(X_train, y_train)
    coeffs=np.empty(len(clf.coef_[0]))
    for i,coef in enumerate(clf.coef_[0]):
        print(f"Feature {i} has coefficent: {coef}")
        coeffs[i]=coef
    return coeffs

#This function visualizes the coefficients effect
def visualize_features_log_reg(X, coeffs):
    feature_importance_df=pd.DataFrame(dict(zip(X.columns,coeffs),index=[0]))
    #Visualize feature Importance
    feature_importance_df.T.sort_values(by=0, ascending=True).plot(kind="bar",legend=False)
    plt.title("Logistic Regression Feature Importance")
    plt.show();

# This function calculates the effect of each Random Forest Claassifier coefficient in the score
def feature_importance_random_forest(X_train, y_train):
    clf_2=RandomForestClassifier(bootstrap= True,max_depth=20,max_features= 'sqrt',min_samples_leaf= 2,min_samples_split= 2,n_estimators=100)
    clf_2.fit(X_train, y_train)
    return np.array(clf_2.feature_importances_)

#This function visualizes the coefficients effect
def visualize_features_random_forest(X,coeffs):
    feature_importance_rf=pd.DataFrame(dict(zip(X.columns,coeffs)),index=[0])
    feature_importance_rf.T.sort_values(by=0, ascending = True).plot(kind="bar",legend=False)
    plt.title("Random Forest Classifier Feature Importance")
    plt.show();