import pandas as pd

#load the data using pandas read from csv and turn it into a dataframe
def load_data():
    data=pd.read_csv("data/heart-disease (2).csv")
    return data
