from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_blobs
import collections

# サンプルデータの作成
X, Y = make_blobs(random_state=8,
                  n_samples=300, 
                  n_features=2, 
                  cluster_std=1.2,
                  centers=2)
print("[X, Y]: ", X)
plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.title('before')
plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# DBSCANの実行
clustering = DBSCAN(eps=1.7, min_samples=4).fit(X)
clustering.labels_
print("クラスタリングラベル: ", clustering.labels_)

count = collections.Counter(clustering.labels_)
print("各クラスタの点の個数: ", count)

plt.subplot(1,2,2)
plt.title('after')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=clustering.labels_, s=50, edgecolor='k')

plt.show()
