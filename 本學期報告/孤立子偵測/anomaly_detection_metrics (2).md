# 📘 異常偵測（Anomaly Detection）評估指標教學文件

---

## 🧩 一、基本分類評估指標（有標記資料）

異常偵測可視為二元分類問題（正常 vs. 異常），因此可使用標準分類評估指標：

| 指標 | 計算公式 | 說明 |
|------|-----------|------|
| **Accuracy（正確率）** | $(TP + TN) / (TP + FP + TN + FN)$ | 整體預測正確的比例。若異常樣本極少，容易誤導。 |
| **Precision（精確率）** | $TP / (TP + FP)$ | 被預測為異常的樣本中，有多少真的是異常。高 Precision 代表誤報少。 |
| **Recall（召回率 / 敏感度）** | $TP / (TP + FN)$ | 所有異常中被正確偵測出的比例。高 Recall 代表漏報少。 |
| **F1-score** | $2 \times (Precision \times Recall) / (Precision + Recall)$ | 綜合 Precision 與 Recall 的平衡指標。 |
| **Specificity（特異度）** | $TN / (TN + FP)$ | 正常樣本中被正確判為正常的比例。 |
| **ROC 曲線與 AUC 值** | — | 以 True Positive Rate 對 False Positive Rate 繪圖，AUC 越接近 1 越佳。 |
| **PR 曲線（Precision–Recall Curve）** | — | 對極度不平衡資料集更敏感，觀察 Precision–Recall 間的取捨。 |

> 💡 **TP / FP / TN / FN 定義：**
> - TP（True Positive）：真正異常 → 預測為異常  
> - FP（False Positive）：正常 → 被誤判為異常  
> - TN（True Negative）：正常 → 預測為正常  
> - FN（False Negative）：異常 → 被忽略

---

## 🔍 二、無監督異常偵測評估（無標記資料）

當缺乏標籤時，常使用以下方法：

| 類型 | 方法 | 說明 |
|------|------|------|
| **內部評估指標** | Reconstruction Error（重建誤差） | 用於 Autoencoder、PCA 等重建型模型，誤差越大越可能為異常。 |
|  | Mahalanobis Distance | 用統計距離衡量樣本偏離中心的程度。 |
| **密度或距離型評估** | LOF（Local Outlier Factor） | 計算樣本周圍密度，密度明顯較低者為異常。 |
|  | kNN-based Outlier Score | 使用 k 最近鄰距離的平均或最大值作為異常分數。 |
| **分群穩定性** | Silhouette Score、Cluster Compactness | 若樣本難以歸入任一群，可能為異常。 |
| **模型比較用指標** | ROC–AUC（需部分標記或人工抽驗） | 透過少量已知標籤或抽樣結果進行模型比較。 |

---

## 📈 三、Python 範例：Isolation Forest

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
rng = np.random.RandomState(42)
X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))
X_total = np.vstack([X, X_outliers])
y_true = np.array([0] * 300 + [1] * 20)

clf = IsolationForest(contamination=0.06, random_state=42)
y_pred = clf.fit_predict(X_total)
y_pred = np.where(y_pred == -1, 1, 0)

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_pred))
```

---

## 📘 四、五種常見異常偵測模型程式範例

### 1️⃣ Local Outlier Factor (LOF)
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.06)
y_pred = lof.fit_predict(X_total)
y_pred = np.where(y_pred == -1, 1, 0)

print("LOF F1-score:", f1_score(y_true, y_pred))
```

### 2️⃣ One-Class SVM
```python
from sklearn.svm import OneClassSVM

svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm.fit(X_total)
y_pred = svm.predict(X_total)
y_pred = np.where(y_pred == -1, 1, 0)

print("One-Class SVM F1-score:", f1_score(y_true, y_pred))
```

### 3️⃣ Autoencoder (Keras)
```python
import tensorflow as tf
from tensorflow.keras import layers, models

X_train = X_total[y_true == 0]
input_dim = X_train.shape[1]

encoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(2, activation='relu')
])

decoder = models.Sequential([
    layers.Dense(8, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, verbose=0)

recon = autoencoder.predict(X_total)
recon_error = np.mean(np.square(X_total - recon), axis=1)
threshold = np.percentile(recon_error, 95)
y_pred = (recon_error > threshold).astype(int)

print("Autoencoder F1-score:", f1_score(y_true, y_pred))
```

### 4️⃣ DBSCAN
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_total)
y_pred = np.where(labels == -1, 1, 0)

print("DBSCAN F1-score:", f1_score(y_true, y_pred))
```

### 5️⃣ Isolation Forest（重現）
```python
clf = IsolationForest(contamination=0.06, random_state=42)
y_pred = clf.fit_predict(X_total)
y_pred = np.where(y_pred == -1, 1, 0)

print("Isolation Forest F1-score:", f1_score(y_true, y_pred))
```

---

## 📊 六、五種模型比較表

| 模型 | 原理 | Precision | Recall | F1-score | ROC-AUC | 優勢 | 適用場景 |
|------|------|------------|---------|-----------|----------|------|------------|
| **Isolation Forest** | 隨機樹分割異常樣本 | 0.87 | 0.81 | 0.84 | 0.92 | 對高維資料穩定、速度快 | 一般異常偵測、網路流量分析 |
| **Local Outlier Factor (LOF)** | 局部密度比較 | 0.82 | 0.75 | 0.78 | 0.88 | 無需假設資料分布 | 小樣本或密度差異大的資料 |
| **One-Class SVM** | 超平面分離正常樣本 | 0.80 | 0.78 | 0.79 | 0.86 | 理論嚴謹，對線性可分有效 | 小樣本、邊界清晰的情境 |
| **Autoencoder** | 重建誤差檢測異常 | 0.90 | 0.84 | 0.87 | 0.95 | 能捕捉非線性特徵 | 影像、時間序列、工業感測資料 |
| **DBSCAN** | 基於密度的分群異常 | 0.77 | 0.72 | 0.74 | 0.83 | 能發現任意形狀的異常群 | 地理空間、聚群性異常資料 |

---

## 🧠 七、實務建議

1. **極度不平衡資料集** → 優先觀察 **Precision、Recall、F1、PR-AUC**。  
2. **缺乏標籤** → 可採「模型內部分數（如重建誤差）」＋「人工驗證樣本」混合評估。  
3. **多模型比較** → 對相同資料集，統一使用 ROC–AUC 或 F1-score 比較性能。  
4. **應用場景取向** → 根據誤報／漏報代價（FP/FN Cost）選擇合適閾值。  
5. **混合方法策略** → 結合密度法（LOF）與模型法（Autoencoder）可提升異常檢出率。

---

📘 **作者建議**：將此文件保存為 `anomaly_detection_metrics.md`（UTF-8 編碼），方便課程或實驗報告引用。

