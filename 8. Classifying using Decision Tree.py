import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("D:\\ML\\Datasets\\ads.csv")
print(data.head())
print(data.isnull().any())
x = data.iloc[:,1:4].values 
y = data.iloc[:,4:5].values
print(x)
print(y[:10])
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x[:,0] = lb.fit_transform(x[:,0]) 
print(x)
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x, y, test_size = 0.1,random_state=0)
print(x_train)
print(x_test[:10])
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
print(dt.fit(x_train,y_train))
y_pred = dt.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
import sklearn.metrics as metrics
fpr, tpr ,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)
plt.plot(fpr,tpr) 
plt.xlim([0,1])
plt.ylim([0,1])
plt.style.use("dark_background")
from sklearn.tree import export_graphviz
export_graphviz(dt, out_file ='tree.dot',
feature_names = ["Gender","Age","Salary"], class_names = ['0','1'],
rounded = True, proportion = False, precision = 2, filled = True)
