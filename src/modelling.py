import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Split the data into X & y

def train_test(data):
    X=data.drop("target", axis=1)
    y=data["target"]

    np.random.seed(42)

    #Split into train and test splits
    X_train, X_test, y_train, y_test= train_test_split(X, y , test_size=0.2)

    #Scaling
    scaler= StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.fit_transform(X_test)
    return X_train, X_test, y_train, y_test,X




#Create a function to fit and score models

def fit_and_score(X_train, X_test, y_train, y_test):
    #Create a dictionary with the models
    models={"Logistic Regression": LogisticRegression(max_iter=100),
    "KNN":KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()}

    #Set random seed
    np.random.seed(42)
    #Make a dictionary to keep model scores
    model_score={}
    for name,model in models.items():
        model.fit(X_train, y_train)
        model_score[name]=model.score(X_test, y_test)
    return model_score

