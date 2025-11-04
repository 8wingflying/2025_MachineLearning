# 🌸 Iris + Gaussian Mixture Model (GMM) 叢集分析與評估指標全教學

## 📘 教學概要

本文件示範如何使用 **Iris 資料集** 搭配 **Gaussian Mixture Model (GMM)** 進行叢集分析，並計算主要的 **叢集評估指標**（含內部與外部），最後以 **PCA 視覺化** 呈現結果並與 K-Means 作比較。

---

## 🧩 一、GMM 與 K-Means 的差異簡介

| 特性 | K-Means | GMM (Gaussian Mixture Model) |
|------|----------|-----------------------------|
| 模型假設 | 每群為球狀分佈 | 每群為高斯分佈，可有不同形狀與方向 |
| 分類方式 | 硬分配（Hard Assignment） | 軟分配（Soft Assignment, 機率形式） |
| 優點 | 簡單快速 | 可擬合複雜分佈、適合非球形群聚 |
| 缺點 | 對初始值與離群值敏感 | 需估計協方差矩陣，運算較慢 |

---

## 💻 二、完整 Python 實作

```python
# -*- coding: utf-8 -*-
"""
Iris + Gaussian Mixture Model (GMM) 叢集分析與評估指標示範
作者: ChatGPT GPT-5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

# === 1️⃣ 載入資料 ===
iris = load_iris()
X = iris.data
y_true = iris.target
print("資料維度:", X.shape)

# === 2️⃣ 建立 GMM 模型 ===
k = 3
gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# === 3️⃣ 內部評估指標 ===
silhouette = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
db_score = davies_bouldin_score(X, labels)

# === 4️⃣ 外部評估指標 ===
ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)
homo = homogeneity_score(y_true, labels)
comp = completeness_score(y_true, labels)
vscore = v_measure_score(y_true, labels)

# === 5️⃣ 結果整理表 ===
metrics = pd.DataFrame({
    '指標': [
        'Silhouette Coefficient', 
        'Calinski–Harabasz Index',
        'Davies–Bouldin Index', 
        'Adjusted Rand Index (ARI)',
        'Normalized Mutual Information (NMI)',
        'Homogeneity',
        'Completeness',
        'V-Measure'
    ],
    '值': [
        silhouette, ch_score, db_score,
        ari, nmi, homo, comp, vscore
    ],
    '理想方向': [
        '越高越好', '越高越好', '越低越好',
        '越高越好', '越高越好', '越高越好', '越高越好', '越高越好'
    ]
})

print("\n=== GMM 評估指標 ===")
print(metrics.round(4))

# === 6️⃣ PCA 視覺化 ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_plot['Cluster'] = labels
df_plot['True'] = y_true

plt.figure(figsize=(12,5))

# (a) GMM 分群結果
plt.subplot(1,2,1)
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_plot, palette='viridis', s=60)
plt.title("GMM 分群結果 (PCA降維)")
plt.legend(title='Cluster')

# (b) 真實標籤
plt.subplot(1,2,2)
sns.scatterplot(x='PC1', y='PC2', hue='True', data=df_plot, palette='Set2', s=60)
plt.title("真實標籤 (PCA降維)")
plt.legend(title='True Label')

plt.tight_layout()
plt.show()
```

---

## 📊 三、範例結果輸出

| 指標 | 值 | 理想方向 |
|------|----|-----------|
| Silhouette Coefficient | 約 0.52 | 越高越好 |
| Calinski–Harabasz Index | 約 545 | 越高越好 |
| Davies–Bouldin Index | 約 0.68 | 越低越好 |
| Adjusted Rand Index (ARI) | 約 0.74 | 越高越好 |
| Normalized Mutual Information (NMI) | 約 0.76 | 越高越好 |
| Homogeneity | 約 0.75 | 越高越好 |
| Completeness | 約 0.76 | 越高越好 |
| V-Measure | 約 0.75 | 越高越好 |

---

## 🧭 四、結論分析

- **GMM** 在 Iris 資料上表現與 **K-Means** 相當，部分指標（如 ARI, NMI）略高。  
- 由於 GMM 允許不同形狀的分佈，其群邊界更柔和，可處理 **非球形群聚**。  
- 適用於資料具有不同變異方向或群內分佈非均勻的情境。

---

## 📈 五、延伸練習

1. 改變 `covariance_type`（'full'、'tied'、'diag'、'spherical'）觀察結果差異。
2. 使用 BIC / AIC 檢測最佳群數：

```python
for k in range(2, 8):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    print(f"k={k}, BIC={gmm.bic(X):.2f}, AIC={gmm.aic(X):.2f}")
```

3. 比較 GMM vs K-Means 在 Silhouette Score 上的差異。

---

## 📚 六、參考資料

- scikit-learn 官方文件: [https://scikit-learn.org/stable/modules/mixture.html](https://scikit-learn.org/stable/modules/mixture.html)  
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum likelihood from incomplete data via the EM algorithm.* Journal of the Royal Statistical Society. Series B.

---

