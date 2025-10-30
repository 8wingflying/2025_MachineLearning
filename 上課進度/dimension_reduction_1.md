## 🧠 降維（Dimension Reduction）教學文件

### 📘 第 1 章：降維的概念與目的

**降維（Dimension Reduction）** 是資料前處理中極為重要的一步，主要目的在於：
- 去除冗餘特徵、降低噪音。
- 減少模型訓練時間與儲存成本。
- 保留資料的主要結構與資訊。
- 有助於資料視覺化（2D / 3D 展示高維資料）。

常見應用：
- 影像特徵壓縮（如 CNN feature embedding）
- NLP 詞嵌入後的可視化（如 Word2Vec + t-SNE）
- 機器學習模型特徵選擇前的前處理

---

### 📊 第 2 章：降維方法分類

| 方法類別 | 名稱 | 特性 |
|-----------|------|------|
| **線性方法** | PCA（主成分分析） | 保留最大方差方向，假設線性關係 |
| | LDA（線性判別分析） | 監督式降維，最大化類別間距離 |
| **非線性方法** | t-SNE | 保留鄰近關係，適合高維資料視覺化 |
| | UMAP | 高速且保持局部與全域結構 |
| | AutoEncoder | 使用神經網路自動學習低維特徵表示 |

---

### 🧩 第 3 章：PCA 主成分分析

**PCA（Principal Component Analysis）** 透過特徵值分解協方差矩陣，找出資料方差最大的方向。

#### 🧮 數學原理
1. 標準化資料。
2. 計算協方差矩陣 \( \Sigma = \frac{1}{n-1} X^T X \)
3. 特徵值分解 \( \Sigma v_i = \lambda_i v_i \)
4. 選取最大 \( k \) 個特徵向量組成投影矩陣。

#### 🧑‍💻 Python 範例
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X = load_iris().data
y = load_iris().target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA on Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

---

### 🧭 第 4 章：LDA 線性判別分析

**LDA（Linear Discriminant Analysis）** 是監督式降維，透過最大化類間散佈與最小化類內散佈達成分類效果最佳的投影。

#### 🧮 目標函數
\[
\max_W \frac{|W^T S_B W|}{|W^T S_W W|}
\]

#### 🧑‍💻 Python 範例
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X = load_iris().data
y = load_iris().target

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='rainbow')
plt.title('LDA on Iris Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()
```

---

### 🌈 第 5 章：t-SNE（t-Distributed Stochastic Neighbor Embedding）

**t-SNE** 是一種非線性降維方法，適合高維資料的視覺化。它會：
- 將高維空間的相似度映射到低維空間。
- 保留鄰近點的相對關係（局部結構）。

#### 🧑‍💻 Python 範例
```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.title('t-SNE on Digits Dataset')
plt.show()
```

---

### 🚀 第 6 章：UMAP（Uniform Manifold Approximation and Projection）

**UMAP** 是近年非常流行的非線性降維方法，能保留局部與全域結構，且速度遠快於 t-SNE。

#### 🧑‍💻 Python 範例
```python
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral')
plt.title('UMAP on Digits Dataset')
plt.show()
```

---

### 🤖 第 7 章：AutoEncoder 非線性降維（深度學習方法）

**AutoEncoder** 是一種無監督式神經網路，用於學習輸入資料的壓縮表示。它包含：
- **Encoder**：將高維資料壓縮成低維潛在向量（latent vector）。
- **Decoder**：從低維潛在空間重建原始資料。

#### 🧠 原理示意
```
Input → [Encoder] → Latent Representation → [Decoder] → Reconstructed Output
```

#### 🧑‍💻 Python 範例（Keras）
```python
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 載入資料
X = load_iris().data

# 定義 AutoEncoder 結構
input_dim = X.shape[1]
encoding_dim = 2  # 壓縮到 2 維

input_layer = Input(shape=(input_dim,))
encoded = Dense(4, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='linear')(encoded)
decoded = Dense(4, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 訓練模型
autoencoder.fit(X, X, epochs=200, batch_size=16, verbose=0)

# 取出 Encoder 部分
encoder = Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X)

# 視覺化低維表示
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=load_iris().target, cmap='viridis')
plt.title('AutoEncoder Nonlinear Dimension Reduction')
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.show()
```

#### ✅ 優點
- 可學習非線性結構。
- 可擴展至深層網路（Deep AutoEncoder）。

#### ⚠️ 缺點
- 訓練成本高、需要大量資料。
- 結果依賴網路設計與初始化。

---

### 🧠 第 8 章：方法比較

| 方法 | 屬性 | 保留局部結構 | 保留全域結構 | 計算速度 | 適用場景 |
|------|------|--------------|--------------|----------|----------|
| PCA  | 線性 | ❌ | ✅ | 🚀 快 | 線性特徵分析 |
| LDA  | 線性（監督） | ✅ | ✅ | 🚀 快 | 類別區分 |
| t-SNE | 非線性 | ✅ | ❌ | 🐢 慢 | 高維資料視覺化 |
| UMAP | 非線性 | ✅ | ✅ | ⚡ 快 | 高維嵌入與聚類 |
| AutoEncoder | 非線性 | ✅ | ✅ | ⚙️ 中 | 深度學習壓縮與特徵萃取 |

---

### 📘 第 9 章：延伸練習建議

1. 比較 PCA、t-SNE、UMAP、AutoEncoder 在 `MNIST` 資料上的降維結果。
2. 探索 **Variational AutoEncoder (VAE)** 作為機率式降維。
3. 將降維後資料輸入 SVM、RandomForest 等分類器，比較準確率差異。
4. 對高維嵌入向量（例如 BERT）使用 UMAP / AutoEncoder 進行可視化。

---

### 🎯 第 10 章：總結

- 降維是機器學習與資料視覺化的重要步驟。
- 線性方法（PCA、LDA）簡單快速，非線性方法（t-SNE、UMAP、AutoEncoder）能捕捉更複雜的結構。
- AutoEncoder 將傳統統計與深度學習結合，成為現代降維的重要工具。

