# 🧠 Ensemble Learning 集成學習套件教學文件

## 一、集成學習概述

**集成學習 (Ensemble Learning)** 是將多個基模型 (Base Learners) 結合以提升模型泛化能力的技術。  
其主要思想是 **「眾人智慧勝於個人」**。

### 常見方法
- **Bagging (裝袋法)**: 並行訓練，降低方差
- **Boosting (提升法)**: 串行訓練，降低偏差
- **Stacking / Blending (疊加法)**: 多層融合，提升泛化性能

---

## 二、Python 主流集成學習套件

| 套件名稱 | 特點 | 適用場景 | 安裝指令 |
|-----------|--------|-------------|------------|
| **scikit-learn** | 基本 Bagging / Boosting / Stacking | 通用教學與實驗 | `pip install scikit-learn` |
| **XGBoost** | 高效 GBDT | 結構化資料 | `pip install xgboost` |
| **LightGBM** | 大數據 & GPU 支援 | 高維資料 | `pip install lightgbm` |
| **CatBoost** | 原生類別特徵支援 | 混合型資料 | `pip install catboost` |
| **mlens** | 專業 Stacking 工具 | 複雜模型融合 | `pip install mlens` |
| **PyCaret** | 自動化 ML | 快速原型設計 | `pip install pycaret` |

---

## 三、常見集成方法與原理

### 1. Bagging (裝袋法)

通過重複抽樣多個子集，各自訓練基模型，最後取平均或投票。

#### 代表演算法：Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 資料載入
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 模型訓練
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 預測與評估
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

sns.heatmap([[acc]], annot=True, cmap="YlGnBu")
plt.title("Random Forest Accuracy")
plt.show()
```

---

### 2. Boosting (提升法)

通過串行訓練，各階段模型重點處理前一階段錯誤樣本。

#### 代表演算法：AdaBoost / XGBoost / LightGBM / CatBoost

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

sns.barplot(x=xgb.feature_importances_, y=range(X.shape[1]), orient='h')
plt.title("XGBoost Feature Importance")
plt.show()
```

---

### 3. Stacking (疊加法)

多模型多層結構：基模型 (Level-0) 預測輸出，依此訓練 Meta Learner (Level-1)。

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

estimators = [
    ('svc', SVC(probability=True)),
    ('dt', DecisionTreeClassifier())
]

stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack_clf.fit(X_train, y_train)
print("Stacking Accuracy:", stack_clf.score(X_test, y_test))
```

---

### 4. Blending (混合法)

使用保留資料集 (Hold-out) 來訓練第二層模型。

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

pred1 = rf.predict_proba(X_test)[:,1]
pred2 = xgb.predict_proba(X_test)[:,1]

blend_X = np.vstack([pred1, pred2]).T
blend_model = LogisticRegression().fit(blend_X, y_test)
print("Blending 模型測試準確率:", blend_model.score(blend_X, y_test))
```

---

## 四、比較表

| 方法 | 是否並行 | 偏差 | 方差 | 可解釋性 | 常見模型 |
|------|-----------|------|------|-------------|-----------|
| Bagging | ✅ | 中 | 低 | 高 | Random Forest |
| Boosting | ❌ | 低 | 中 | 中 | XGBoost / LightGBM |
| Stacking | ❌ | 低 | 低 | 中 | Multi-Model |
| Blending | ✅ | 低 | 低 | 中 | Logistic Meta |

---

## 五、視覺化比較

```python
models = ['RandomForest', 'XGBoost', 'Stacking']
scores = [rf.score(X_test, y_test),
          xgb.score(X_test, y_test),
          stack_clf.score(X_test, y_test)]

sns.barplot(x=models, y=scores, palette="viridis")
plt.title("不同集成模型準確率比較")
plt.show()
```

---

## 六、優缺點分析

| 方法 | 優點 | 缺點 |
|------|------|------|
| **Bagging** | 降低方差，穩定性高 | 解釋性下降 |
| **Boosting** | 高準確，處理複雜關係 | 易過擬合，訓練時間長 |
| **Stacking** | 高彈性，能融合異質模型 | 設計複雜，需大量資料 |
| **Blending** | 實現簡單，效果良好 | Hold-out 資料浪費 |

---

## 七、學習資源

- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [LightGBM Guide](https://lightgbm.readthedocs.io)
- [CatBoost User Guide](https://catboost.ai)
- [PyCaret Docs](https://pycaret.gitbook.io/docs)

---

## 八、結論

集成學習是模型性能提升的核心技術，無論是分類、回歸或異常偵測，都能得到顯著改善。

> ☀️ **小提示**: 結合「模型多樣性 + 加權優化 + 正則化」，才能發揮集成學習最大效益。

