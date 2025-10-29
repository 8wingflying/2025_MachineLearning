# 🧠 Ensemble Learning Cheatsheet

## 📘 四大主流集成學習策略
| 類型 | 中文名稱 | 核心概念 | 常見模型 | 主要優勢 |
|------|-----------|-----------|------------|-----------|
| **Bagging** | 裝袋法 | 多模型在隨機取樣的資料上訓練後平均 | RandomForest | 穩定、抗過擬合 |
| **Boosting** | 提升法 | 逐步修正前一模型的錯誤 | AdaBoost / XGBoost / LightGBM | 高準確率、可調整權重 |
| **Stacking** | 堆疊法 | 不同模型輸出組合成次級模型 | StackingClassifier | 模型互補、靈活性高 |
| **Blending** | 混合法 | 用驗證集加權融合模型預測 | 自定 Logistic Regression | 簡單快速、適合 Kaggle |

---

## ⚙️ 常用 Python 套件
| 套件名稱 | 功能 | 備註 |
|-----------|------|------|
| `scikit-learn` | Bagging / Boosting / Stacking | 最完整的集成方法集合 |
| `xgboost` | 高效梯度提升樹 | GPU 支援、業界標準 |
| `lightgbm` | 快速梯度提升 | 適合大型表格資料 |
| `catboost` | 類別特徵友好 | 無需 One-Hot 編碼 |
| `mlens` / `vecstack` | Stacking / Blending 工具 | sklearn 相容擴充 |

---

## 🔢 Bagging 範例（Random Forest）
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
rf_model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])
rf_model.fit(X_train, y_train)
```

---

## 🚀 Boosting 範例
### AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier
ada_model = Pipeline([
    ("preprocess", preprocess),
    ("model", AdaBoostClassifier(n_estimators=200, learning_rate=0.8, random_state=42))
])
```

### XGBoost
```python
import xgboost as xgb
xgb_model = Pipeline([
    ("preprocess", preprocess),
    ("model", xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42))
])
```

### LightGBM
```python
import lightgbm as lgb
lgb_model = Pipeline([
    ("preprocess", preprocess),
    ("model", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42))
])
```

---

## 🧠 Stacking 範例
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('xgb', xgb.XGBClassifier(n_estimators=300)),
]
stack_model = Pipeline([
    ("preprocess", preprocess),
    ("model", StackingClassifier(estimators=base_models, final_estimator=LogisticRegression()))
])
```

---

## 🔄 Blending 範例
```python
from sklearn.linear_model import LogisticRegression
rf_model.fit(X_tr, y_tr)
xgb_model.fit(X_tr, y_tr)
val_rf = rf_model.predict_proba(X_val)[:,1]
val_xgb = xgb_model.predict_proba(X_val)[:,1]
blend_X_val = np.vstack([val_rf, val_xgb]).T
blender = LogisticRegression().fit(blend_X_val, y_val)
```

---

## 📊 評估與視覺化 (Matplotlib + Seaborn)
```python
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

models = {'RF': y_pred_rf, 'XGB': y_pred_xgb, 'LGB': y_pred_lgb}
acc = [accuracy_score(y_test, y_pred) for y_pred in models.values()]
sns.barplot(x=list(models.keys()), y=acc, palette='viridis')
plt.title('Accuracy Comparison')
plt.show()
```

---

## 🧩 快速建議對照表
| 場景 | 推薦方法 | 套件 |
|------|------------|------|
| 小型表格資料 | Bagging / Stacking | sklearn |
| 大型表格資料 | Boosting | XGBoost / LightGBM |
| 自動化實驗 | AutoML | PyCaret / AutoGluon |
| 深度學習融合 | Blending | TensorFlow / PyTorch + sklearn |

---

✅ **速查提示**：
- Bagging 穩定模型、Boosting 提升準確率
- Stacking + Blending 為高階融合策略
- XGBoost 幾乎是所有 Kaggle 比賽的核心武器