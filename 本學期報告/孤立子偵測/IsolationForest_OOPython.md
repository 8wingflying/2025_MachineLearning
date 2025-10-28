#

## 實作
- 三個主要類別：
  - IsolationTree：負責建構單棵孤立樹
  - IsolationForest：負責整體森林訓練與預測
  - DemoApp：示範如何使用此模型進行異常偵測
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# ==========================================
# 1️⃣ Isolation Tree
# ==========================================
class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.size = None

    def fit(self, X, current_height=0):
        self.size = X.shape[0]

        # 終止條件
        if current_height >= self.height_limit or self.size <= 1:
            return self

        # 隨機選擇特徵與分割值
        feature = np.random.randint(0, X.shape[1])
        min_val, max_val = np.min(X[:, feature]), np.max(X[:, feature])
        if min_val == max_val:
            return self
        split_value = np.random.uniform(min_val, max_val)

        # 分割資料
        left_mask = X[:, feature] < split_value
        X_left, X_right = X[left_mask], X[~left_mask]

        self.split_feature = feature
        self.split_value = split_value
        self.left = IsolationTree(self.height_limit)
        self.right = IsolationTree(self.height_limit)
        self.left.fit(X_left, current_height + 1)
        self.right.fit(X_right, current_height + 1)
        return self

    def path_length(self, x, current_height=0):
        if self.left is None or self.right is None:
            return current_height + c_factor(self.size)
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)


# ==========================================
# 2️⃣ Isolation Forest
# ==========================================
def c_factor(n):
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n  # 調整常數

class IsolationForest:
    def __init__(self, n_estimators=100, sample_size=256):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X):
        self.trees = []
        height_limit = int(np.ceil(np.log2(self.sample_size)))
        for _ in range(self.n_estimators):
            sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
            tree = IsolationTree(height_limit)
            tree.fit(sample)
            self.trees.append(tree)
        return self

    def anomaly_score(self, X):
        paths = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            path_lengths = np.array([tree.path_length(x) for tree in self.trees])
            paths[i] = np.mean(path_lengths)
        c = c_factor(self.sample_size)
        scores = np.power(2, -paths / c)
        return scores

    def predict(self, X, threshold=0.5):
        scores = self.anomaly_score(X)
        return np.where(scores >= threshold, -1, 1)  # -1: 異常, 1: 正常


# ==========================================
# 3️⃣ Demo 範例
# ==========================================
class DemoApp:
    @staticmethod
    def run():
        X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)
        X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
        X_total = np.vstack([X, X_outliers])

        model = IsolationForest(n_estimators=100, sample_size=128)
        model.fit(X_total)

        scores = model.anomaly_score(X_total)
        labels = model.predict(X_total, threshold=0.6)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_total[:, 0], X_total[:, 1], c=labels, cmap='coolwarm')
        plt.title("Isolation Forest (OOP Implementation)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()


if __name__ == "__main__":
    DemoApp.run()
```
