# 🌳 Decision Tree 教學文件（以 Python 物件導向實作）

---

## 🧠 一、決策樹基本概念

決策樹（Decision Tree）是一種**監督式學習演算法**，可用於分類（Classification）或回歸（Regression）問題。  
其核心思想是：
> 根據特徵的資訊增益（Information Gain）或基尼指數（Gini Index）進行遞迴式分裂，形成樹狀結構模型。

---

## 📊 二、關鍵概念

| 名稱 | 說明 | 常用公式 |
|------|------|-----------|
| 熱 (Entropy) | 衡量資料的不確定性 | `H(D) = -Σ p_i log2(p_i)` |
| 資訊增益 (Information Gain) | 分裂後不確定性的減少量 | `Gain(D, A) = H(D) - Σ (|D_v|/|D|) * H(D_v)` |
| 基尼指數 (Gini Index) | 衡量分類純度 | `Gini(D) = 1 - Σ (p_i)^2` |

---

## 🧬 三、Python 物件導向實作範例

### 📘 範例說明
此範例使用 OOP 方式自建一個簡化版的 Decision Tree，用遞迴實作分類邏輯。

```python
import numpy as np
from collections import Counter

class DecisionNode:
    """決策樹節點類別"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # 若為葉節點，儲放最終類別

class DecisionTreeClassifierOOP:
    """以物件導向方式實作簡化版決策樹"""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_entropy = self.entropy(y)

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left_idx = X[:, feature_idx] <= t
                right_idx = X[:, feature_idx] > t
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                left_entropy = self.entropy(y[left_idx])
                right_entropy = self.entropy(y[right_idx])
                child_entropy = (len(y[left_idx]) / len(y)) * left_entropy + \
                                (len(y[right_idx]) / len(y)) * right_entropy

                info_gain = parent_entropy - child_entropy
                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx, split_thresh = feature_idx, t
        return split_idx, split_thresh

    def build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth:
            most_common = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=most_common)

        feature_idx, threshold = self.best_split(X, y)
        if feature_idx is None:
            most_common = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=most_common)

        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold
        left_child = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionNode(feature_idx, threshold, left_child, right_child)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, x, node=None):
        node = node or self.root
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(sample) for sample in X])
```

---

## 🧪 四、測試與比較

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 載入 Iris 資料集
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用自建的 OOP 決策樹
tree = DecisionTreeClassifierOOP(max_depth=3)
tree.fit(X_train, y_train)
y_pred_custom = tree.predict(X_test)

# 使用 scikit-learn 的 DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
sk_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
sk_tree.fit(X_train, y_train)
y_pred_sklearn = sk_tree.predict(X_test)

# 準確率比較
print("自建 OOP 決策樹準確率:", accuracy_score(y_test, y_pred_custom))
print("scikit-learn 決策樹準確率:", accuracy_score(y_test, y_pred_sklearn))
```

---

## 🌈 五、視覺化（使用 scikit-learn）

```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
tree.plot_tree(sk_tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
```

---

## 🧠 六、測驗題庫（含解析）

| 題號 | 題目 | 類型 | 答案 | 解析 |
|------|------|------|------|------|
| 1 | 決策樹是屬於哪一類型的機器學習方法？ | 單選 | 監督式學習 | 因為有標籤資料 (y) 用來訓練分類模型。 |
| 2 | 熱越高代表資料的不確定性如何？ | 單選 | 越高 | 熱衡量系統的不確定性，越高代表資料越混亂。 |
| 3 | 資訊增益的公式為何？ | 單選 | H(D) - Σ(|D_v|/|D|)H(D_v) | 衡量分裂後的不確定性減少量。 |
| 4 | 基尼指數越低代表？ | 單選 | 節點越純 | 純度越高表示分類效果越好。 |
| 5 | 在分類問題中，決策樹的葉節點代表？ | 單選 | 最終的分類結果 | 葉節點儲存最終預測的類別值。 |

---

## 💾 七、延伸閱讀
- Quinlan, J. R. (1986). *Induction of decision trees.*
- Scikit-learn 官方文件: https://scikit-learn.org/stable/modules/tree.html
- NIST AI RMF 與解釋性模型分析

---

> 📦 **作者註**：本文件以 OOP 實作說明 Decision Tree 的基本運作邏輯，可擴充至支援 Gini、CART 或 ID3。  
> 適合教育與模型內部解釋性分析使用。

