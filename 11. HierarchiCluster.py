import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('D:\\ML\\Datasets\\Mall_Customers.csv')
print(dataset.head()) 
print(dataset.isnull().any())
x = dataset.iloc[:, [3, 4]].values
print(x[:5])
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward')) 
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances') 
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=100, color='red', label = 'Cluster 1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=100, color='blue', label = 'Cluster 2')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], s=100, color='green', label = 'Cluster 3')
plt.scatter(x[y_hc==3,0], x[y_hc==3,1], s=100, color='cyan', label = 'Cluster 4')
plt.scatter(x[y_hc==4,0], x[y_hc==4,1], s=100, color='magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.legend()
plt.show()
