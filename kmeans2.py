import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv("./iris.csv")
data = data.drop(["Species", "Id"], axis=1)

kmeans = KMeans(n_clusters= 3)
kmeans.fit(data.values)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ["g.", "r.", "c."]

print(centroids)
print(labels)

for i in range (len(data)) :
    # print (labels[i])
    # print (colors[labels[i]])
    plt.plot(data.values[i][0] * data.values[i][1], 
                data.values[i][2] * data.values[i][3], 
                colors[labels[i]], markersize =10)

plt.scatter(centroids[:,0], centroids[:,1])
plt.show()