'''
Silhouette Score is one of the most widely used clustering evaluation metrics, it helps answer a simple quesion

How good is my cluster ?

What Makes a good cluster?

1. Cohesion(Compactness)(Points within the same cluster should be close.)
2. Separation(cluster should be far from other cluseter)

a = Average distance to points within its own clustering.(a should be smaller)
b = Average distance to nearest neighboring clustering.(b should be larger)
'''

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

X,y = make_blobs(
    n_samples=500,
    centers=3,
    random_state=42
)

#apply kmeans
model = KMeans(
    n_clusters=3,
    random_state=42
)

labels = model.fit_predict(X)

print("silhouette_score: ",silhouette_score(X,labels))


plt.figure(figsize=(8,8))
plt.scatter(X[:,0],X[:,1],c=labels)
plt.title("KMeans Clustering")
plt.show()



'''
0.8 - 1 -> Excellent Clustering
0.51 - 0.8 -> Good clustering
0.3 - 0.5 -> average clustering
below this -> poor clutering

'''