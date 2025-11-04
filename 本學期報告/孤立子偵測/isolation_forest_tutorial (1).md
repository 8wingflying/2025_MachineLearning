# 🌲 Isolation Forest 教學文件

## 1️⃣ 模型概述

**Isolation Forest**（孤立森林）是一種以「孤立（Isolation）」概念為核心的異常值偵測（Outlier Detection）演算法。  
它由 Liu et al.（2008）提出，專門設計來快速偵測資料中少數異常樣本。

### 🧠 核心概念
- 正常資料點 → 分佈集中 → 需要較多次切割才能被孤立  
- 異常資料點 → 分佈稀疏 → 少數幾次隨機切割即可被孤立

因此，Isolation Forest 透過「隨機樹」的結構，利用樣本被孤立所需的平均路徑長度（path length）來評估是否為異常。

---

## 2️⃣ 基本原理

### 📘 隨機切割樹（Isolation Tree）
每棵樹的建構方式如下：
1. 隨機選取一個特徵。
2. 在該特徵範圍內，隨機選擇一個分割點。
3. 重複步驟 1–2，直到每個節點只包含一筆資料或達到最大深度。

### 📊 平均路徑長度（Average Path Length）
對於每個樣本 \( x \)，計算其在森林中被孤立所需的平均路徑長度 \( E(h(x)) \)。

異常分數（Anomaly Score）定義如下：

\[
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
\]

其中：
- \( E(h(x)) \)：樣本在所有樹的平均路徑長度  
- \( c(n) \)：n 節點的平均路徑長度常數  
  \[
  c(n) = 2H(n-1) - \frac{2(n-1)}{n}
  \]
  \( H(i) \) 為調和數。

異常分數越接近 1 → 越可能為異常。

---

## 3️⃣ Python 實作範例

### 📦 匯入必要模組
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
```

### 📊 生成模擬資料
```python
# 產生正常資料
X, _ = make_blobs(n_samples=300, centers=[[0, 0], [5, 5]], cluster_std=0.8, random_state=42)

# 增加異常點
outliers = np.random.uniform(low=-6, high=10, size=(20, 2))
X = np.vstack((X, outliers))
```

### 🌲 建立 Isolation Forest 模型
```python
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X)

# 預測結果：1=正常，-1=異常
y_pred = clf.predict(X)
scores = clf.decision_function(X)
```

### 🔍 視覺化結果
```python
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='coolwarm', s=30, edgecolor='k')
plt.title("Isolation Forest 異常偵測結果")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

## 4️⃣ 重要參數解釋

| 參數名稱 | 說明 |
|-----------|------|
| `n_estimators` | 森林中樹的數量，數量越多穩定性越高但速度變慢 |
| `max_samples` | 每棵樹的樣本數，可設為 `"auto"`（預設 = min(256, n_samples)） |
| `contamination` | 預期的異常比例（例如 0.05 表示約 5% 為異常） |
| `max_features` | 每棵樹選取的特徵數 |
| `random_state` | 隨機種子，確保結果可重現 |

---

## 5️⃣ 實務應用範例

### 📢 案例：信用卡交易異常偵測
```python
df = pd.read_csv("creditcard.csv")
X = df.drop(columns=["Class"])  # 特徵
y = df["Class"]  # 標籤 (0=正常, 1=異常)

clf = IsolationForest(contamination=0.02, random_state=42)
clf.fit(X)
y_pred = clf.predict(X)
y_pred = np.where(y_pred == -1, 1, 0)

from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
```

---

## 6️⃣ 超參數調整建議

| 目標 | 調整方向 |
|------|-----------|
| 提高異常偵測靈敏度 | 減少 `contamination` 或增加 `n_estimators` |
| 降低誤判 | 提高 `contamination` 或限制 `max_samples` |
| 加快速度 | 減少 `n_estimators` 或 `max_samples` |

---

## 7️⃣ 常見問題（FAQ）

**Q1. 與 One-Class SVM 有何不同？**  
- Isolation Forest 速度快、可擴展至大型資料集。  
- One-Class SVM 計算成本高但邊界更精準。  

**Q2. 可用於非數值資料嗎？**  
需先將類別特徵進行編碼（如 One-Hot Encoding）。

**Q3. 適合高維資料嗎？**  
是的，Isolation Forest 對高維度相對穩定，適合如網路封包、工業感測器資料。

---

## 8️⃣ 延伸練習

1. 嘗試更改 `contamination` 值觀察異常比例變化。  
2. 以 3D 資料視覺化異常偵測結果。  
3. 結合 PCA 降維後再進行 Isolation Forest。  
4. 應用於 IoT 資料流或網路安全日誌的即時異常監控。

---

## 9️⃣ 參考文獻
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. IEEE ICDM.  
- Scikit-learn 官方文件：[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

---

> 📘 **總結**：  
> Isolation Forest 是一種以隨機化為核心、具高效能與可擴展性的異常值偵測方法。  
> 對於工業監控、金融詐騙、網路入侵偵測等應用領域都相當實用。

