import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importiing the dataset
dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 22)
