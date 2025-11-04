# 🌸 Iris 資料集無監督學習（Unsupervised Clustering）教學

---

## 📘 一、目標說明
無監督學習的目的在於 **不使用標籤資料（species）** 的情況下，讓模型自行尋找資料內部的結構與分群關係。  
本章將以 **K-Means 分群** 與 **階層式分群 (Hierarchical Clustering)** 為主，分析 Iris 資料集的三種花博系統結構。

---

## 🧩 二、載入資料與前處理

```python
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 載入資料
iris = sns.load_dataset("iris")
X = iris.drop(columns="species")

# 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## ⚙️ 三、K-Means 分群分析

### 1️⃣ 計算不同 k 值下的 SSE（肝部法則 Elbow Method）

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sse = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.plot(K, sse, 'bx-')
plt.xlabel('群數 k')
plt.ylabel('SSE (誤差平方和)')
plt.title('Elbow Method for Optimal k')
plt.show()
```

📈 **觀察重點：**
- SSE 在 k=3 之後下降趨勢明顯變緩 → 最佳分群數為 **3 群**。

---

### 2️⃣ 建立 K-Means 模型並視覺化結果

```python
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

kmeans = KMeans(n_clusters=3, random_state=42)
iris["cluster"] = kmeans.fit_predict(X_scaled)

# PCA 降維至2D視覺化
pca = PCA(n_components=2)
iris_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=['PCA1', 'PCA2'])
iris_pca["cluster"] = iris["cluster"]

plt.figure(figsize=(8,6))
sns.scatterplot(x="PCA1", y="PCA2", hue="cluster", data=iris_pca, palette="Set2", s=80)
plt.title("K-Means 分群結果 (PCA 2D 視覺化)")
plt.show()
```

---

### 3️⃣ 與真實標籤比較 (使用 Adjusted Rand Index)

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(iris["species"], iris["cluster"])
print(f"Adjusted Rand Index (ARI): {ari:.3f}")
```

📊 **結果解讀：**
- ARI 越接近 1，代表分群結果越接近真實標籤。  
- 在 Iris 資料集中，通常 ARI 約為 **0.7~0.8**，代表分群效果良好。

---

## 🦤 四、階層式分群 (Hierarchical Clustering)

### 1️⃣ 以 Scipy 套件繪製樹狀圖 (Dendrogram)

```python
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=(10, 6))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram (ward linkage)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()
```

📘 **說明：**
- `ward` 方法最常用，用於最小化群內變異。
- 從樹狀圖可視覺化分群層級，並觀察適合的群數（約 3 群）。

---

### 2️⃣ 以 AgglomerativeClustering 建立模型

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
iris["agg_cluster"] = agg.fit_predict(X_scaled)

# PCA 視覺化
iris_pca["agg_cluster"] = iris["agg_cluster"]

sns.scatterplot(x="PCA1", y="PCA2", hue="agg_cluster", data=iris_pca, palette="Set1", s=80)
plt.title("階層式分群結果 (PCA 2D 視覺化)")
plt.show()
```

---

## 📚 五、結果比較與觀察

| 方法 | 概念 | 優點 | 缺點 |
|------|------|------|------|
| **K-Means** | 基於距離最小化 | 高效、常用 | 對初始值與 k 敏感 |
| **階層式分群** | 基於層級距離合併 | 可視化層次結構 | 不易擴展至大量資料 |
| **Iris 實驗結果** | k≈3 可有效區分三類花 | Setosa 清楚、其他兩類略有重疊 |

---

## 🚀 六、延伸方向

1. 使用 **DBSCAN** 或 **Gaussian Mixture Model (GMM)** 進一步比較分群效果。  
2. 分析不同距離度量（例如 cosine distance）。  
3. 探討分群後各群的特徵平均差異（群中心分析）。

---

## 📆 七、Python 套件需求

```bash
pip install pandas seaborn matplotlib scikit-learn scipy
```

---

🗕 **建立日期：** 2025-10-28  
✍️ **作者：** ChatGPT 教學助手  
🧠 **主題：** Unsupervised Clustering Analysis on Iris Dataset  

