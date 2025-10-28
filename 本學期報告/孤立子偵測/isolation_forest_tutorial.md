# 🧠 Isolation Forest 教學文件

## 1️⃣ Isolation Forest 的定義與用途

**Isolation Forest（孤立森林）** 是一種基於「孤立原理」(Isolation Principle) 的異常偵測演算法。  
它假設異常資料點（outliers）更容易被「孤立」於一般資料，因此可藉由隨機切割空間來區分異常與正常樣本。

📘 **主要用途：**
- 偵測金融交易異常（如詐騙偵測）
- 網路入侵與安全事件分析
- 機器感測資料異常監控（IoT）
- 生產線設備故障預測
- 異常客戶行為分析（例如消費行為偏離群體）

---

## 2️⃣ 工作原理（原理解析）

Isolation Forest 的核心概念在於「隨機切割資料空間以孤立點」。

### (1) 核心概念
- 利用隨機選取的特徵與分割值，遞迴切割資料（形成隨機樹）。
- 當樣本越容易被孤立（越快被分開），其異常程度越高。
- 計算每個樣本的平均孤立路徑長度 (path length)，再轉換為異常分數。

### (2) 數學原理
異常分數定義如下：

\[
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
\]

其中：
- \(E(h(x))\)：樣本 x 的平均孤立深度（在多棵樹中的平均分割層數）
- \(c(n)\)：樣本數 n 下的平均路徑長度期望值
- 分數 \(s(x, n)\) 越接近 1，代表該樣本越可能是異常點

---

## 3️⃣ 實作步驟與參數解釋

### ✅ 實作步驟

1. **資料準備：**  
   準備含有可能異常值的資料集。  
2. **建立模型：**  
   使用 `sklearn.ensemble.IsolationForest`。
3. **模型訓練：**  
   呼叫 `.fit()` 方法對資料進行學習。
4. **預測異常：**  
   使用 `.predict()` 或 `.decision_function()` 取得異常分數。
5. **結果視覺化：**  
   繪製異常點分布圖。

---

### ⚙️ 主要參數說明

| 參數名稱 | 說明 | 常見預設值 |
|-----------|------|------------|
| `n_estimators` | 樹的數量（越多越穩定但耗時） | 100 |
| `max_samples` | 每棵樹抽樣的樣本數，可設為 `"auto"` 或數字 | `"auto"` |
| `contamination` | 資料中預期的異常比例，用於決定閾值 | 0.1 |
| `max_features` | 每棵樹隨機選取的特徵比例 | 1.0 |
| `bootstrap` | 是否使用抽樣重複（bootstrap） | False |
| `random_state` | 隨機種子以重現結果 | None |

---

## 4️⃣ Isolation Forest 的優缺點分析

| 類別 | 說明 |
|------|------|
| ✅ 優點 | - 適用於高維度資料<br>- 不需標記異常樣本（非監督式）<br>- 計算速度快，記憶體需求低<br>- 可處理大型資料集 |
| ⚠️ 缺點 | - 對資料分佈敏感（若資料非對等或偏態，結果可能不穩定）<br>- 當異常點與正常點距離不明顯時準確率下降<br>- 不易直接解釋模型結果 |

---

## 5️⃣ Python 實作範例

以下範例展示如何在 Python 中使用 `IsolationForest` 進行異常偵測。

### 📦 套件安裝

```bash
pip install scikit-learn matplotlib numpy pandas
```

---

### 🧩 範例程式碼

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. 產生模擬資料
rng = np.random.RandomState(42)
X_inliers = 0.3 * rng.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]  # 正常群
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))  # 異常點
X = np.r_[X_inliers, X_outliers]

# 2. 建立 Isolation Forest 模型
clf = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
clf.fit(X)

# 3. 預測
y_pred = clf.predict(X)
scores = clf.decision_function(X)

# 4. 結果視覺化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k')
plt.title("Isolation Forest 異常偵測結果")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 5. 檢視異常比例
anomaly_ratio = np.sum(y_pred == -1) / len(y_pred)
print(f"異常樣本比例：{anomaly_ratio:.2f}")
```

---

### 📊 輸出結果解讀

- `predict()` 回傳：
  - `1` → 正常樣本
  - `-1` → 異常樣本
- `decision_function()` → 返回異常分數（越小越異常）

---

### 🗂️ 實驗資料集建議
若要使用真實資料，可嘗試：
- [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [UCI Machine Learning Repository - Anomaly Datasets](https://archive.ics.uci.edu/)

---

## 6️⃣ 結論與應用建議

Isolation Forest 是異常偵測中高效且直覺的方法，適用於以下場景：
- 大量無標筆資料的異常行為分析  
- 資料維度高且異常比例低的場景  
- 作為預警或偵測模型的前置篩選  

若需要提升偵測效能，可考慮：
- 調整 `contamination` 參數以匹配真實異常比