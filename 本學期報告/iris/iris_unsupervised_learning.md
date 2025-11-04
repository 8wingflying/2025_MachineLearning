# 🌸 Iris 資料集的非監督學習教學  
（含 K-Means・Hierarchical・DBSCAN・PCA・t-SNE・UMAP・LDA・GMM）

---

## 📘 一、前言

非監督學習（Unsupervised Learning）方法於無需標籤的情況下，用來探索資料的內在結構。
在 Iris 資料集中，我們會應用多種分群與降維演算法，
並加入 **LDA (監督式對照)** 與 **GMM (高斯混合模型)** 進行對照比較。

---

## 📊 二、資料載入與標準化
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target   # 僅用於 LDA 對照

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🌼 三、K-Means 分群
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

scores = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

plt.plot(range(2, 7), scores, marker='o')
plt.title("Silhouette Score vs Cluster Number")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels_km, palette='viridis')
plt.title("K-Means Clustering (k=3)")
plt.show()
```

---

## 🌿 四、層次式分群 (Hierarchical Clustering)
```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

labels_h = fcluster(Z, 3, criterion='maxclust')
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels_h, palette='rainbow')
plt.title("Hierarchical Clustering (Ward linkage)")
plt.show()
```

---

## 🌻 五、DBSCAN 分群
```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.8, min_samples=5)
labels_db = db.fit_predict(X_scaled)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels_db, palette='coolwarm')
plt.title("DBSCAN Clustering")
plt.show()
```

---

## 🌺 六、PCA 線性降維
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_km, palette='viridis')
plt.title("K-Means on PCA-reduced Iris Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

print("\u89e3\u91cb\u8b8a\u7570\u6bd4:", pca.explained_variance_ratio_)
```

---

## 🌈 七、t-SNE 非線性降維
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_km, palette='Spectral')
plt.title("t-SNE Visualization of Iris Clusters")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
```

---

## 🌸 八、UMAP 非線性降維
```python
import umap

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels_km, palette='cool')
plt.title("UMAP Visualization of Iris Clusters")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
```

---

## 🌼 九、LDA (監督式) 降維對照
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=iris.target, palette='Set2')
plt.title("LDA Projection of Iris (Supervised Reference)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.show()
```

---

## 🌷 十、GMM (高斯混合模型)
```python
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)

sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels_gmm, palette='viridis')
plt.title("Gaussian Mixture Model Clustering")
plt.show()

ari = adjusted_rand_score(y, labels_gmm)
print(f"Adjusted Rand Index (\u8207\u771f\u5be6\u6a19\u7c64\u76f8\u4f3c\u5ea6): {ari:.3f}")
```

---

## 🌺 十一、在降維空間 (PCA、UMAP) 上套用 GMM 分群

### ✪ 1. PCA + GMM 可視化
```python
gmm_pca = GaussianMixture(n_components=3, random_state=42)
labels_gmm_pca = gmm_pca.fit_predict(X_pca)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_gmm_pca, palette='Set1')
plt.title("GMM Clustering on PCA-reduced Iris Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
```

### ✪ 2. UMAP + GMM 可視化
```python
gmm_umap = GaussianMixture(n_components=3, random_state=42)
labels_gmm_umap = gmm_umap.fit_predict(X_umap)

sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels_gmm_umap, palette='Set2')
plt.title("GMM Clustering on UMAP-reduced Iris Data")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
```

---

## 🌻 十二、方法比較總表

| 方法 | 類型 | 是否需標籤 | 特性 | 分群形狀 | 優點 | 缺點 |
|------|------|-----------|------|-----------|------|------|
| K-Means | 分群 | 否 | 以中心點最小化群內距離 | 球狀 | 簡單、快速 | 對初始點敏感 |
| Hierarchical | 分群 | 否 | 依距離合併 | 任意 | 可視化層次 | 大樣本效率低 |
| DBSCAN | 分群 | 否 | 密度基礎 | 任意形 | 自動離群點 | 參數敏感 |
| GMM | 分群 | 否 | 模擬模型 | 橢圓形 | 可輸出機率 | 易陷局部最小值 |
| PCA | 降維 | 否 | 線性 | 全局 | 瞭解資料變異 | 無法分群 |
| t-SNE | 降維 | 否 | 非線性 | 局部 | 分群視覺清楚 | 不保全局結構 |
| UMAP | 降維 | 否 | 非線性 | 局部 + 全局 | 快速、穩定 | 參數敏感 |
| LDA | 降維 | 是 | 類別最大化 | 線性 | 類別分離清楚 | 需標籤 |

---

## 🌼 十三、延伸挑戰
1. 比較 GMM 與 K-Means 在 **ARI / NMI** 指標下的差異。  
2. 將 GMM 群的等高線線 (Gaussian Contour) 繪刻在 PCA 空間中。  
3. 在 t-SNE 降維空間中嘗試 GMM 並比較群體穩定性。  
4. 使用 GMM 的機率輸出進行 soft label 訓練。  
5. 嘗試 Bayesian GMM (使用 `BayesianGaussianMixture`) 自動推斷群數。  

---
```

