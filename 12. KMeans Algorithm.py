import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('D:\\ML\\Datasets\\Mall_Customers.csv') 
print(dataset.head())
print(dataset.isnull().any())
x = dataset.iloc[:,[3,4]].values
print(x[:5])
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300,
                    n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) 
plt.plot(range(1,11),wcss) 
plt.title('The Elbow Method') 
plt.xlabel('Number of clusters') 
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5, init = 'k-means++', max_iter=300, n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans[:5])
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],s=100,c='red',label='cluster 1') 
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1],s=100,c='blue',label='cluster 2') 
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1],s=100,c='green',label='cluster 3') 
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1],s=100,c='yellow',label='cluster 4') 
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1],s=100,c='brown',label='cluster 5') 
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='centroids') 
plt.title('Clusters of clients')
plt.xlabel("Annual Income in 1000 $") 
plt.ylabel("Spending Score (1-1000") 
plt.legend()
plt.show()
