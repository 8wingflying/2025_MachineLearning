## Iris資料集的特徵工程
```
載入資料集
特徵縮放 (StandardScaler)
類別編碼 (One-Hot Encoding)
特徵選取 (SelectKBest)
降維 (PCA)
```

````python
"""
Iris 資料集 特徵工程流程示範
步驟:
1. 載入資料
2. 特徵縮放 (StandardScaler)
3. 類別編碼 (OneHotEncoder)
4. 特徵選取 (SelectKBest)
5. 降維 (PCA)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# 1. 載入資料集
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("原始資料前 5 筆：")
print(df.head())

# 2. 特徵縮放 (數值標準化)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n標準化後的前 5 筆：")
print(X_scaled[:5])

# 3. 類別編碼 (One-Hot Encoding) — 對目標變數
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

print("\nOne-Hot 編碼後的目標變數前 5 筆：")
print(y_encoded[:5])

# 4. 特徵選取 (SelectKBest, 使用 ANOVA F-test)
selector = SelectKBest(score_func=f_classif, k=2)  # 只挑選 2 個最重要的特徵
X_selected = selector.fit_transform(X_scaled, y)

print("\n挑選出來的重要特徵形狀：", X_selected.shape)

# 5. PCA 降維 (降到 2 維)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA 降維後的前 5 筆：")
print(X_pca[:5])

```
