
## kmeans vs GaussianMixture
```python
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# 創建資料，並視覺化
X, y = datasets.make_blobs(n_samples=1500,cluster_std=[1.0, 2.5, 0.5], random_state=170)

plt.figure(figsize=(12,4))
plt.rcParams['font.family'] = 'STKaiti'
plt.rcParams['font.size'] = 20
plt.subplot(1,3,1)
plt.scatter(X[:,0],X[:,1],c = y)
plt.title('Original Data',pad = 20)


kmeans = KMeans(3)
kmeans.fit(X)
y_ = kmeans.predict(X)
plt.subplot(1,3,2)
plt.scatter(X[:,0],X[:,1],c = y_)
plt.title('KMeans',pad = 20)

gmm = GaussianMixture(n_components=3)
y_ = gmm.fit_predict(X)
plt.subplot(1,3,3)
plt.scatter(X[:,0],X[:,1],c = y_)
plt.title('GMM',pad = 20)
 
plt.figtext(x = 0.51,y = 1.1,s = 'KMeans VS GMM',ha = 'center',fontsize = 30)
plt.savefig('./GMM高斯混合模型.png',dpi = 200)
```
