import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data/psoriasis/data.csv")
excel = pd.read_excel("data/psoriasis/scores.xlsx")
print(data.head())

# make a list of the median of each column

medians = []

for column in data.columns:
    median = data[column].median()
    medians.append(median)