import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('D:\\ML\\Datasets\\store_data.csv',header = None) 
print(dataset.head())
x = dataset.iloc[:, 0:].values
print(x)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
transactions[:2]
from apyori import apriori
rules = apriori(transactions, min_support = 0.0045, min_confidence = 0.2, min_lift = 3, min_length=2)
results = list(rules)
print(results)
