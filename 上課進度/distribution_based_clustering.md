# Distribution-based Clustering 教學文件
*Version: v2 （含 Gaussian Mixture Model 實作與分布視覺化圖範例）*

---

## 📘 一、分布式分群（Distribution-based Clustering）簡介

**Distribution-based Clustering（分布式分群）** 是以「機率分布模型」為基礎的分群方法，假設資料是由多個不同的機率分布（如高斯分布）所生成。透過估計這些分布的參數，能夠找出隱藏的群集結構。

常見演算法：
- Gaussian Mixture Model (**GMM**)
- Expectation-Maximization (**EM Algorithm**)

---

## 🧩 二、理論基礎

### 1️⃣ 主要思想
假設資料集 \( X = \{x_1, x_2, ..., x_n\} \) 是由 \( K \) 個分布產生，則整體分布可表示為：

\[
P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
\]

其中：
- \( \pi_k \)：第 \( k \) 個分布的混合權重（\( \sum \pi_k = 1 \)）
- \( \mu_k \)：第 \( k \) 個高斯分布的平均值
- \( \Sigma_k \)：第 \( k \) 個高斯分布的共變異數矩陣

---

## ⚙️ 三、Expectation-Maximization（EM）演算法步驟

1. **E 步驟（Expectation）**：根據當前參數，計算每個樣本屬於各分布的機率。
2. **M 步驟（Maximization）**：更新分布參數，使得觀察資料的似然值最大化。
3. **重複 E-M** 直到收斂。

---

## 🧮 四、Gaussian Mixture Model（GMM）數學公式

高斯分布的機率密度函數：

\[
\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) \right)
\]

整體模型為加權和：

\[
P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
\]

---

## 🧠 五、Python 實作（Gaussian Mixture Model）

```python
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

# 生成樣本資料
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 建立 GMM 模型
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X)
labels = gmm.predict(X)

# 繪圖
plt.figure(figsize=(7,5))
sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='deep', s=40)
plt.title("Gaussian Mixture Model Clustering Result")
plt.show()
```

---

## 📈 六、GMM 分布視覺化圖（Matplotlib 輸出）

### ✅ 使用橢圓可視化高斯分布範圍

```python
import numpy as np
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle, width, height = 0, 2 * np.sqrt(covariance), 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
for pos, covar in zip(gmm.means_, gmm.covariances_):
    draw_ellipse(pos, covar, alpha=0.3, color='red')
plt.title('Gaussian Mixture Model - Distribution Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

📊 **輸出解釋：**
- 每個紅色橢圓代表一個高斯分布的範圍。
- 點顏色代表所屬群集。

---

## 📊 七、GMM 與 K-Means 比較

| 特性 | GMM | K-Means |
|------|------|---------|
| 模型假設 | 機率分布 | 幾何距離 |
| 分群邊界 | 軟分群（Soft） | 硬分群（Hard） |
| 支援橢圓形群集 | ✅ 是 | ❌ 否 |
| 離群點處理 | 較佳 | 一般 |
| 輸出結果 | 每點的群集機率 | 群集標籤 |

---

## 🧩 八、應用案例

- 金融風險模型中的客群分層
- 語音辨識中的聲學模型
- 圖像分割（例如膚色區域偵測）
- 文本主題分群（以詞嵌入後套用 GMM）

---

## 🧮 九、優缺點整理

| 優點 | 缺點 |
|------|------|
| 可建模任意形狀群集 | 需指定群數 |
| 提供機率式分群結果 | 對初始值敏感 |
| 可視化可理解度高 | 高維資料運算量大 |

---

## 📚 十、延伸練習

1. 嘗試不同的 `covariance_type`（`full`, `tied`, `diag`, `spherical`）。
2. 將 GMM 套用於降維後的資料（例如 PCA 結果）。
3. 與 DBSCAN 比較分群效果。
4. 在異常偵測任務中利用 GMM 的機率閾值作為判斷依據。

---

## 🧾 十一、參考資料

- scikit-learn 官方文件：[https://scikit-learn.org/stable/modules/mixture.html](https://scikit-learn.org/stable/modules/mixture.html)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective.* MIT Press.

---

📦 檔案名稱建議：
```
DISTRIBUTION_based_Clustering.md
```

📦 執行環境：
```bash
pip install scikit-learn matplotlib seaborn numpy
```

