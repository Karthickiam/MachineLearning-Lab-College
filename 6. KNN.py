import numpy as n
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("D:\\ML\\Datasets\\ads.csv")
print(data.head())
print(data.isnull().any())
x=data.iloc[:,1:4].values
y=data.iloc[:,4:5].values
print(x[:5])
print(y[:5])
from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
x[:,0]= lb.fit_transform(x[:,0])
print(x[:5])
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x, y, test_size = 0.1,random_state=0)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier( n_neighbors = 5 , p = 2 )
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
fpr, tpr ,threshold = metrics.roc_curve(y_test,y_pred) 
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)
plt.plot(fpr,tpr)
plt.xlim([0,1])
plt.ylim([0,1])
plt.style.use("dark_background")
