import pandas as pd

#load the data using pandas read from csv and turn it into a dataframe
def load_data():
    data=pd.read_csv("heart-disease (2) (1).csv")
    return data
