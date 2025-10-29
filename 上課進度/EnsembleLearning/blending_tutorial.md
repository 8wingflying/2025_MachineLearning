# 🤓 混合泛化 (Blending) 教學文件

## 1️⃣ 混合泛化簡介

**混合泛化（Blending）** 是一種 **集成學習 (Ensemble Learning)** 技術，用於結合多個基模型（Base Models）的預測結果，以提升模型整體表現。  
與 Stacking 類似，Blending 也引入一個「次級模型 (Meta Model)」，但訓練方式略有不同：

| 比較項目 | Stacking | Blending |
|-----------|-----------|-----------|
| 第二層資料來源 | 交叉驗證預測 (out-of-fold prediction) | 留出驗證集 (hold-out set) 預測 |
| 訓練集分割 | K-Fold | 訓練集 / 驗證集 分割 |
| 計算量 | 高 (多次交叉驗證) | 較低 |
| 過擬合風險 | 低 | 稍高 (取決於驗證集大小) |

---

## 2️⃣ 原理概述

假設我們有三個基模型：
- 模型 1：`LinearRegression`
- 模型 2：`RandomForestRegressor`
- 模型 3：`GradientBoostingRegressor`

步驟：

1. 訓練集和驗證集分割
2. 訓練基模型
3. 用驗證集產生 meta-features
4. 輸入 meta model 訓練
5. 最終預測

$$
\hat{y} = f_{\text{meta}}(f_1(x), f_2(x), ..., f_n(x))
$$

---

## 3️⃣ Python 實作：迴歸版

(全部程式碼與圖表同上版)

---

## 4️⃣ 分類版 (Classification Blending)

### 4.1 資料

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4.2 基模型

```python
models = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42))
]
for _, model in models:
    model.fit(X_train, y_train)
```

### 4.3 Meta features

```python
meta_features = np.column_stack([
    model.predict_proba(X_valid)[:, 1] for _, model in models
])
meta_model = LogisticRegression()
meta_model.fit(meta_features, y_valid)
final_prob = meta_model.predict_proba(meta_features)[:, 1]
final_pred = (final_prob > 0.5).astype(int)
```

### 4.4 評估

```python
acc = accuracy_score(y_valid, final_pred)
auc = roc_auc_score(y_valid, final_prob)
print(f"Blending Accuracy: {acc:.4f}")
print(f"Blending AUC: {auc:.4f}")
```

---

## 5️⃣ K-Fold Blending 進階技巧

### 原理

使用 K-Fold Cross-Validation 產生 Out-of-Fold (OOF) 預測，可減少驗證集的價值損失。

### 5.1 Python 程式

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
models = [
    ("lr", LinearRegression()),
    ("rf", RandomForestRegressor(random_state=42)),
    ("gb", GradientBoostingRegressor(random_state=42))
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_features = np.zeros((len(X), len(models)))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X[train_idx], X[valid_idx]
    y_tr, y_va = y[train_idx], y[valid_idx]

    for i, (_, model) in enumerate(models):
        model.fit(X_tr, y_tr)
        meta_features[valid_idx, i] = model.predict(X_va)

meta_model = LinearRegression()
meta_model.fit(meta_features, y)
```

---

## 6️⃣ 優缺點

| 優點 | 缺點 |
|------|------|
| 計算效率高 | 驗證集效應敏感 |
| 易懂易用 | 高相關基模型效果不佳 |
| 可強化模型多樣性 | 需經驗證比例設計 |

---

## 7️⃣ 實務建議

- 驗證比例 20~30%
- 多樣性基模型 (線性 + 樹型)
- Meta model 可用 XGBoost / LightGBM
- 分析基模型相關性

---

## 8️⃣ 結論

Blending 是 **高效、快速、精準的集成模型方法**，如果資料量大，可考慮使用 **Stacking (Cross-Validation)** 形式以提升泛化性能。

---

## 📚 後續讀物

- Wolpert, D. H. (1992). *Stacked Generalization*. Neural Networks.  
- Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*.  
- scikit-learn Docs: https://scikit-learn.org/stable/

