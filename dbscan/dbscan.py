from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_blobs
import collections

# サンプルデータの作成
X, Y = make_blobs(random_state=8,
                  n_samples=1000, 
                  n_features=5, 
                  cluster_std=1.8,
                  centers=5)
print("[X, Y]: ", X)

# クラスタリング前の散布図描画
plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.title('before')
plt.scatter(X[:, 0], X[:, 1], marker='o', s=80, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# DBSCANの実行
clustering = DBSCAN(eps=4.6, min_samples=8).fit(X)
print("クラスタリングラベル: ", clustering.labels_)

# 各クラスタの点の個数をカウント
count = collections.Counter(clustering.labels_)
print("各クラスタの点の個数: ", count)

# クラスタリング後の散布図描画
plt.subplot(1,2,2)
plt.title('after')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=clustering.labels_, s=80, edgecolor='k')
plt.show()
