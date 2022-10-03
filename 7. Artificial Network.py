import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('D:\\ML\\Datasets\\Churn_Modelling.csv') 
print(dataset.head())
print(dataset.isnull().any())
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
print(X[:5])
print(y[:5])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.compose import ColumnTransformer
labelencoder_X_1 =LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 =LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])
ct = ColumnTransformer([("Surname", OneHotEncoder(),[1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]
print("X->{}".format(X))
print('\n')
print("y->{}".format(y))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer= "uniform", activation = "relu", input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dense(units = 1, kernel_initializer= "uniform", activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)
from sklearn.metrics import accuracy_score
print("Accuracy score",accuracy_score(y_test,y_pred)*100,"%")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
