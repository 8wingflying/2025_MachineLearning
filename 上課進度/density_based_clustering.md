# Density-based Clustering 教學文件
*Version: v2 (含 DBSCAN × OPTICS × HDBSCAN 圖形輸出範例)*

---

## 📘 一、密度式分群簡介

密度式分群是一種根據資料密度區域來辨識群集的方法，不需預先指定群數，能自動辨識任意形狀的群集與離群點。

### 常見演算法
- DBSCAN
- OPTICS
- HDBSCAN

---

## 🧩 二、DBSCAN 理論與公式

| 名稱 | 定義 |
|------|------|
| ε (epsilon) | 鄰域半徑 |
| MinPts | 最少鄰居數 |

DBSCAN 將資料點分為：核心點、邊界點、離群點。

$$
N_\varepsilon(p) = \{ q \in D \mid dist(p, q) \leq \varepsilon \}
$$

---

## ⚙️ 三、DBSCAN 實作範例

```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

plt.figure(figsize=(7,5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=40)
plt.title("DBSCAN Clustering Result")
plt.show()
```

---

## 🔍 四、OPTICS 實作

```python
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
labels_optics = optics.fit_predict(X)

plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=labels_optics, cmap='Spectral', s=40)
plt.title("OPTICS Clustering Result")
plt.show()
```

---

## 🎗️ 五、HDBSCAN 實作

```python
import hdbscan
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=300, noise=0.06, random_state=42)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels_hdb = clusterer.fit_predict(X)

plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=labels_hdb, cmap='rainbow', s=40)
plt.title("HDBSCAN Clustering Result")
plt.show()
```

---

## 🤷️‍♂️ 六、K-distance Plot

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(X)
distances, _ = nbrs.kneighbors(X)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.title("K-distance Graph (for estimating eps)")
plt.xlabel("Points sorted by distance")
plt.ylabel("k-distance")
plt.show()
```

---

## 🧠 七、演算法比較

| 特性 | DBSCAN | OPTICS | HDBSCAN |
|------|---------|---------|---------|
| 需指定 ε | ☑️ | ❌ | ❌ |
| 自動判群數 | ❌ | ☑️ | ☑️ |
| 可處理變化密度 | 一般 | ☑️ | ☑️ |
| 離群處理 | ☑️ | ☑️ | ☑️ |
| 運算速度 | 快 | 較慢 | 中等 |

---

## 🧩 八、應用範例

- 📍 地理資料分析
- 🚀 遙測圖像分區
- 🧠 異常值偵測
- 💬 社群分析

---

## 🤊 九、優缺點

| 優點 | 缺點 |
|------|------|
| 不需預知群數 | 參數敏感 |
| 能處理離群點 | 高維效果不好 |

---

## 📚 十、延伸練習

1. 使用 `make_circles` 、`make_blobs`測試不同資料分佈。  
2. 將 PCA 降維應用於 DBSCAN。  
3. 將 ε 、MinPts 作 Grid Search觀察效果。  
4. 比較 OPTICS 與 DBSCAN 的差異。

---

## 🗾️ 參考資源

- [scikit-learn DBSCAN 文件](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [HDBSCAN Official Docs](https://hdbscan.readthedocs.io)
- Ester, M., Kriegel, H. P., Sander, J., Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.*

---

## 📂 基本環境設定

```bash
pip install scikit-learn matplotlib hdbscan numpy
```

推薦儲存為：
```
Density_based_Clustering.md
```

