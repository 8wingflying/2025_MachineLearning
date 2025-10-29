# 📘 Blending vs Stacking 教學文件（繁體中文＋英文對照＋圖表＋Python 實作）

---

## 🤠 1. 導論｜Introduction

### 中文：
在集成學習（Ensemble Learning）中，**Blending（混合泛化）** 與 **Stacking（堆疊泛化）** 都屬於「多模型融合」技術。  
它們透過結合多個基模型（Base Models）的預測結果來提升最終模型的泛化能力。  
兩者的主要差異在於「如何利用訓練資料」以及「如何產生第二層模型的輸入」。

### English:
In ensemble learning, **Blending** and **Stacking** are both meta-learning techniques for combining multiple base models to enhance generalization.  
Their main difference lies in how they handle training data and generate meta-model inputs.

---

## 🥉 2. 基本概念比較｜Core Concepts Comparison

| 項目 | **Stacking（堆疊泛化）** | **Blending（混合泛化）** |
|------|----------------------------|----------------------------|
| **主要原理 / Core Idea** | 使用 K-Fold Cross-Validation 建立 out-of-fold 預測，訓練 meta-model。 | 使用一部分驗證集 (holdout set) 來訓練 meta-model。 |
| **資料利用方式 / Data Usage** | 全部訓練資料皆被使用（透過交叉驗證）。 | 一部分資料留作驗證，未用於基模型訓練。 |
| **泛化能力 / Generalization** | 高（因交叉驗證避免過擬合）。 | 較易 overfit（若 holdout 太小）。 |
| **訓練速度 / Speed** | 較慢（多次訓練）。 | 較快（一次訓練即可）。 |
| **實作難度 / Complexity** | 較高 | 較低 |
| **適用場景 / Use Case** | 需要最強泛化能力的情境（如競賽最終模型）。 | 快速測試或小型資料集。 |

---

## ⚙️ 3. 運作流程圖解｜Workflow Diagrams

### ASCII 示意：
```
Stacking (with K-Fold CV):              Blending (with Holdout Set):

 ┌──────────┌                       ┌──────────┌
 │ Train Data │                       │ Train Data │
 └────┬────┘                       └────┬────┘
       │                                   │
     ┌─▼─┐                             ┌───▼───┐
     │K-Fold│                         │Train+Val│
     └─┬─┬─┘                             └───┬───┘
       │                                   │
  Base Models                          Base Models
       │                                   │
  Out-of-Fold Preds                 Holdout Predictions
       │                                   │
     Meta Model → Final Pred         Meta Model → Final Pred
```

---

### 🥮 Matplotlib 視覺化流程圖程式碼

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 5))

# 區塊顏色
train_color = "#AED6F1"
model_color = "#A9DFBF"
meta_color = "#F9E79F"

# Stacking 區域
ax.add_patch(mpatches.Rectangle((0.1, 0.6), 0.3, 0.25, color=train_color))
ax.text(0.25, 0.72, "Training Data\n(K-Fold)", ha='center', fontsize=10)

ax.add_patch(mpatches.Rectangle((0.45, 0.6), 0.25, 0.25, color=model_color))
ax.text(0.57, 0.72, "Base Models\n(out-of-fold preds)", ha='center', fontsize=10)

ax.add_patch(mpatches.Rectangle((0.75, 0.6), 0.15, 0.25, color=meta_color))
ax.text(0.83, 0.72, "Meta Model", ha='center', fontsize=10)

ax.arrow(0.4, 0.725, 0.05, 0, head_width=0.03, head_length=0.03, fc='k', ec='k')
ax.arrow(0.7, 0.725, 0.05, 0, head_width=0.03, head_length=0.03, fc='k', ec='k')
ax.text(0.5, 0.85, "Stacking Flow", fontsize=12, weight='bold')

# Blending 區域
ax.add_patch(mpatches.Rectangle((0.1, 0.15), 0.3, 0.25, color=train_color))
ax.text(0.25, 0.27, "Training + Holdout Data", ha='center', fontsize=10)

ax.add_patch(mpatches.Rectangle((0.45, 0.15), 0.25, 0.25, color=model_color))
ax.text(0.57, 0.27, "Base Models\n(on Train)", ha='center', fontsize=10)

ax.add_patch(mpatches.Rectangle((0.75, 0.15), 0.15, 0.25, color=meta_color))
ax.text(0.83, 0.27, "Meta Model\n(on Holdout)", ha='center', fontsize=10)

ax.arrow(0.4, 0.25, 0.05, 0, head_width=0.03, head_length=0.03, fc='k', ec='k')
ax.arrow(0.7, 0.25, 0.05, 0, head_width=0.03, head_length=0.03, fc='k', ec='k')
ax.text(0.5, 0.4, "Blending Flow", fontsize=12, weight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.show()
```

---

## 🥮 4. Python 實作示例｜Python Implementation

### （A）Stacking 實作（scikit-learn）
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 資料載入
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基模型
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
]

# 元模型
meta_model = LogisticRegression()
stack_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# 訓練與預測
stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)

print("Stacking 準確率:", accuracy_score(y_test, y_pred))
```

---

### （B）Blending 實作（自訂 holdout）
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# 資料分割：train / holdout / test
X, y = load_breast_cancer(return_X_y=True)
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
X_holdout_train, X_test, y_holdout_train, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# 產生 holdout 預測
holdout_pred = np.column_stack([
    rf.predict_proba(X_holdout_train)[:,1],
    gb.predict_proba(X_holdout_train)[:,1]
])

meta = LogisticRegression()
meta.fit(holdout_pred, y_holdout_train)

# 測試階段
test_pred = np.column_stack([
    rf.predict_proba(X_test)[:,1],
    gb.predict_proba(X_test)[:,1]
])
final_pred = meta.predict(test_pred)

print("Blending 準確率:", accuracy_score(y_test, final_pred))
```

---

## 🗾 5. 優缺點對照表｜Pros & Cons

| 項目 | **Stacking** | **Blending** |
|------|---------------|---------------|
| 泛化能力 | ✅ 高（因交叉驗證） | ⚠️ 中等，視 holdout 大小 |
| 計算成本 | ❌ 高 | ✅ 低 |
| 訓練速度 | ❌ 較慢 | ✅ 快速 |
| 實作簡易度 | ❌ 複雜 | ✅ 簡單 |
| 資料利用率 | ✅ 高 | ❌ 部分資料未用於訓練 |
| 過擬合風險 | ✅ 低 | ⚠️ 較高 |
| 適合資料量 | 大型資料 | 小型資料或快速實驗 |

---

## 🦯 6. 使用建議｜Best Practice Recommendations

| 條件 | 建議使用 |
|------|-----------|
| 資料量大且目標是最終高精度模型 | **Stacking** |
| 需要快速測試模型組合

