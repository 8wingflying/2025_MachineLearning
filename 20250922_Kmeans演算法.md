# 20250922  Kmeans演算法
- Googl colab ==>!pip list
- [常用資料集](資料集.md)
- [SCIKIT-LEARN六大套件說明](https://scikit-learn.org/stable/index.html)
  - `1`.
  - `2`.
  - `3`.
  - `4`.
  - `5`.
  - `6`.     
- Kmeans演算法
  - 演算法敘述
  - 演算法實作
  - 使用scikit-learn套件
    - [sklearn.cluster 叢集演算法](https://scikit-learn.org/stable/api/sklearn.cluster.html#module-sklearn.cluster)
    - [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
      - 須說明Parameters:參數的使用
      - 須說明Attributes:屬性的使用
      - 說明重要方法
        - fit(X, y=None, sample_weight=None)
        - fit_predict(X, y=None, sample_weight=None)
        - fit_transform(X, y=None, sample_weight=None) 
  - 評估指標 ==> silhouette_score

## chatgpt產生的程式
- PROMPT:請用python程式寫一個簡單的kmeans

```PYTHON
import numpy as np
import matplotlib.pyplot as plt

# K-means 算法實現
def kmeans(X, k, max_iters=100, tol=1e-4):
    # 隨機初始化 K 個中心點
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iters):
        # 計算每個點與 K 個中心點的距離
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        
        # 將每個點分配給最近的中心點
        labels = np.argmin(distances, axis=1)
        
        # 計算每個群集的新中心點
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 檢查中心點是否收斂
        if np.linalg.norm(new_centers - centers) < tol:
            break
        
        centers = new_centers
    
    return centers, labels

# 生成一些隨機數據
np.random.seed(42)
X = np.vstack([np.random.normal(loc, 0.5, (100, 2)) for loc in [(1, 1), (5, 5), (9, 1)]])

# 執行 K-means
k = 3
centers, labels = kmeans(X, k)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
## chatgpt產生的程式
- PROMPT:請用python程式寫一個簡單的kmeans++
- 當然！K-means++ 是 K-means 算法的一個改進版本，它在選擇初始中心點時會更加智能，能夠有效地減少隨機初始化中心點所帶來的問題。
- K-means++ 的選擇過程是基於距離的概率分佈來選擇下一個中心點，這樣可以使得初始中心點更具代表性。
```PYTHON
import numpy as np
import matplotlib.pyplot as plt

# 計算兩個點之間的歐式距離
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# 使用 K-means++ 初始化選擇初始中心
def initialize_centers(X, k):
    centers = []
    # 隨機選擇一個初始中心
    centers.append(X[np.random.choice(X.shape[0])])
    
    for _ in range(1, k):
        # 計算每個點到最近中心的距離的平方
        dist_sq = np.min(np.array([euclidean_distance(x, center)**2 for center in centers for x in X]).reshape(X.shape[0], len(centers)), axis=1)
        
        # 根據距離的平方來進行概率選擇新的中心點
        prob = dist_sq / dist_sq.sum()
        cumulative_prob = np.cumsum(prob)
        random_value = np.random.rand()
        
        # 根據累積概率選擇下一個中心
        for j, p in enumerate(cumulative_prob):
            if random_value < p:
                centers.append(X[j])
                break
                
    return np.array(centers)

# K-means++ 算法實現
def kmeans_plus_plus(X, k, max_iters=100, tol=1e-4):
    # 使用 K-means++ 初始化中心點
    centers = initialize_centers(X, k)
    
    for i in range(max_iters):
        # 計算每個點到所有中心點的距離
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        
        # 為每個點分配最接近的中心點
        labels = np.argmin(distances, axis=1)
        
        # 計算新的中心點
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 檢查收斂條件
        if np.linalg.norm(new_centers - centers) < tol:
            break
        
        centers = new_centers
        
    return centers, labels

# 生成一些隨機數據
np.random.seed(42)
X = np.vstack([np.random.normal(loc, 0.5, (100, 2)) for loc in [(1, 1), (5, 5), (9, 1)]])

# 執行 K-means++
k = 3
centers, labels = kmeans_plus_plus(X, k)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title("K-means++ Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

```
import numpy as np
import matplotlib.pyplot as plt

# 計算兩個點之間的歐式距離
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# 使用 K-means++ 初始化選擇初始中心
def initialize_centers(X, k):
    centers = []
    # 隨機選擇一個初始中心
    centers.append(X[np.random.choice(X.shape[0])])
    
    for _ in range(1, k):
        # 計算每個點到最近中心的距離的平方
        dist_sq = np.min(np.array([euclidean_distance(x, center)**2 for center in centers for x in X]).reshape(X.shape[0], len(centers)), axis=1)
        
        # 根據距離的平方來進行概率選擇新的中心點
        prob = dist_sq / dist_sq.sum()
        cumulative_prob = np.cumsum(prob)
        random_value = np.random.rand()
        
        # 根據累積概率選擇下一個中心
        for j, p in enumerate(cumulative_prob):
            if random_value < p:
                centers.append(X[j])
                break
                
    return np.array(centers)

# K-means++ 算法實現
def kmeans_plus_plus(X, k, max_iters=100, tol=1e-4):
    # 使用 K-means++ 初始化中心點
    centers = initialize_centers(X, k)
    
    for i in range(max_iters):
        # 計算每個點到所有中心點的距離
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        
        # 為每個點分配最接近的中心點
        labels = np.argmin(distances, axis=1)
        
        # 計算新的中心點
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 檢查收斂條件
        if np.linalg.norm(new_centers - centers) < tol:
            break
        
        centers = new_centers
        
    return centers, labels

# 生成一些隨機數據
np.random.seed(42)
X = np.vstack([np.random.normal(loc, 0.5, (100, 2)) for loc in [(1, 1), (5, 5), (9, 1)]])

# 執行 K-means++
k = 3
centers, labels = kmeans_plus_plus(X, k)

# 可視化結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title("K-means++ Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
