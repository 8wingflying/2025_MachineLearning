---
title: "imbalanced-learn 教學文件"
author: "T Ben"
date: 2025-10-29
---

# imbalanced-learn 教學文件 ｜ Imbalanced-learn Tutorial
*(含 Matplotlib + Seaborn 視覺化實作與重點摘要)*

---

## 📘 一、模組介紹 ｜ Introduction

`imbalanced-learn`（簡稱 `imblearn`）是一個專門解決「資料不平衡（Imbalanced Dataset）」問題的 Python 套件，常與 `scikit-learn` 一起使用。

### ✅ Key Takeaways
- `imbalanced-learn` 主要用於平衡類別資料分佈。
- 可與 sklearn pipeline 無縫整合。
- 提供多種過採樣與欠採樣技術。

---

## ⚖️ 二、資料不平衡的定義 ｜ What is Data Imbalance?

當某類別樣本的數量遠少於其他類別（例如 95% vs 5%）時，分類模型容易偏向多數類別，導致錯誤預測。

| 類型 | 說明 | 範例 |
|------|------|------|
| 類別比例失衡 | 某類樣本過少 | 信用卡詐欺 (fraud: 1%, normal: 99%) |
| 標籤稀有事件 | 罕見事件難以偵測 | 工業設備故障預測 |
| 多分類偏態 | 某幾類樣本佔據大多數 | 多疾病分類資料集 |

### ✅ Key Takeaways
- 當資料不平衡時，Accuracy 不再可靠。
- 模型會傾向預測多數類別。

---

## 🧠 三、主要解決策略 ｜ Common Strategies

1. **權重調整 (Class Weighting)**  
2. **欠採樣 (Under-sampling)**  
3. **過採樣 (Over-sampling)**  
4. **混合採樣 (Hybrid Sampling)**  

### ✅ Key Takeaways
- 權重調整適合中等不平衡問題。
- SMOTE 與 ADASYN 是常見的過採樣方法。

---

## 🧰 四、常用方法 ｜ Common imblearn Methods

| 類別 | 方法 | 說明 |
|------|------|------|
| 欠採樣 | `RandomUnderSampler` | 隨機刪除多數類樣本 |
| 過採樣 | `RandomOverSampler` | 複製少數類樣本 |
| 過採樣 | `SMOTE` | 使用 KNN 生成新樣本 |
| 過採樣 | `ADASYN` | 針對難分類樣本合成新資料 |
| 過採樣 | `KMeansSMOTE` | 利用聚類資訊產生更平衡的新樣本 |
| 混合 | `SMOTEENN`, `SMOTETomek` | 結合過採樣與清理策略 |

### ✅ Key Takeaways
- SMOTE 是最常用的生成方法。
- KMeansSMOTE 對高維度資料表現較穩定。

---

## 💻 五、Python 實作 ｜ Basic Implementation

### (1) 安裝套件
```bash
pip install imbalanced-learn seaborn matplotlib scikit-learn
```

### (2) 建立不平衡資料集
```python
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                           weights=[0.9, 0.1], random_state=42)
print(Counter(y))

sns.countplot(x=y)
plt.title("Before Resampling")
plt.show()
```

### ✅ Key Takeaways
- 使用 `make_classification` 快速建立測試資料集。
- 可使用 Seaborn 繪製類別分佈圖。

---

## 🔄 六、SMOTE 過採樣示例 ｜ SMOTE Example
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(Counter(y_res))

sns.countplot(x=y_res)
plt.title("After SMOTE Resampling")
plt.show()
```

### ✅ Key Takeaways
- SMOTE 可生成合成樣本，避免單純重複資料。
- 有助於提升少數類模型識別率。

---

## ⚙️ 七、ADASYN 與 KMeansSMOTE 比較 ｜ ADASYN vs KMeansSMOTE

```python
from imblearn.over_sampling import ADASYN, KMeansSMOTE

ada = ADASYN(random_state=42)
X_ada, y_ada = ada.fit_resample(X, y)

kms = KMeansSMOTE(random_state=42)
X_kms, y_kms = kms.fit_resample(X, y)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
sns.countplot(x=y, ax=axs[0]); axs[0].set_title("Original")
sns.countplot(x=y_ada, ax=axs[1]); axs[1].set_title("ADASYN")
sns.countplot(x=y_kms, ax=axs[2]); axs[2].set_title("KMeansSMOTE")
plt.tight_layout()
plt.show()
```

### ✅ Key Takeaways
- ADASYN 更注重生成「難分類」樣本。
- KMeansSMOTE 結合聚類結果以產生更平衡的資料。

---

## 🧩 八、與模型整合 ｜ Model Integration (Pipeline)
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### ✅ Key Takeaways
- 可將 SMOTE 與模型整合進同一 Pipeline。
- 避免資料洩漏（Data Leakage）。

---

## 📊 九、性能比較圖 ｜ Performance Comparison
```python
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'KMeansSMOTE': KMeansSMOTE(random_state=42)
}

scores = {}
for name, sampler in methods.items():
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    scores[name] = (f1, auc)

f1s = [v[0] for v in scores.values()]
aucs = [v[1] for v in scores.values()]

fig, ax = plt.subplots(1,2, figsize=(10,4))
sns.barplot(x=list(scores.keys()), y=f1s, ax=ax[0])
ax[0].set_title("F1 Score Comparison")

sns.barplot(x=list(scores.keys()), y=aucs, ax=ax[1])
ax[1].set_title("ROC-AUC Comparison")

plt.tight_layout()
plt.show()
```

### ✅ Key Takeaways
- 使用 F1-score 與 ROC-AUC 衡量少數類別表現。
- 不同方法適合不同資料型態。

---

## ⚙️ 十、延伸章節：XGBoost / LightGBM 調參技巧 ｜ Advanced Tuning

### XGBoost 範例
```python
import xgboost as xgb
clf = xgb.XGBClassifier(
    scale_pos_weight=9,
    eval_metric='aucpr',
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    random_state=42
)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
```

### LightGBM 範例
```python
import lightgbm as lgb
clf = lgb.LGBMClassifier(
    scale_pos_weight=9,
    objective='binary',
    metric='auc',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    random_state=42
)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
```

### ✅ Key Takeaways
- XGBoost / LightGBM 可使用 `scale_pos_weight` 調整樣本權重。
- 選用 `aucpr` 評估指標適合不平衡情境。

---

## ✅ 十一、總結 ｜ Summary
- `imbalanced-learn` 提供多樣化的資料平衡方法。
- 可輕鬆整合至 sklearn pipeline。
- 可結合 XGBoost / LightGBM 進行進階應用。

---

## 📚 參考資源 ｜ References
- [imbalanced-learn 官方文件](https://imbalanced-learn.org/stable/)
- [scikit-learn 官方文件](https://scikit-learn.org/stable/)
- [XGBoost 官方文件](https://xgboost.readthedocs.io/)
- [LightGBM 官方文件](https://lightgbm.readthedocs.io/)

---

