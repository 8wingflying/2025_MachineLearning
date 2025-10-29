# Ensemble Learning 集成學習完整教學  
（以 Titanic 生存預測為案例）

## 目錄
1. 什麼是集成學習（Ensemble Learning）
2. 為什麼集成學習有效？偏差-變異 (Bias-Variance) 直覺
3. 四大主流集成策略總覽
4. Titanic 資料介紹與前處理
5. Bagging：Random Forest
6. Boosting：AdaBoost / XGBoost / LightGBM
7. Stacking：多模型堆疊學習
8. Blending：混合泛化 (hold-out blending)
9. 模型評估與視覺化（Accuracy / 混淆矩陣 / ROC）
10. AutoML 與深度學習集成延伸
11. 實務建議與總結

---

## 1. 什麼是集成學習（Ensemble Learning）

集成學習（Ensemble Learning）是一種**將多個模型結合**來提升預測表現的技術。

> 🧠 概念：多個「弱學習器」（如淺層決策樹）合在一起可形成一個「強學習器」。

舉例：三位醫生各自診斷 → 多數決可降低誤判率。

---

## 2. 為什麼集成學習有效？Bias-Variance 直覺

模型誤差由三部分構成：

- **Bias（偏差）**：模型太簡單，導致系統性誤差。
- **Variance（變異）**：模型太複雜，對訓練資料太敏感。
- **Noise（噪音）**：資料中不可預測的隨機性。

| 方法 | 降低偏差 | 降低變異 |
|------|----------|----------|
| Bagging | ❌ | ✅ |
| Boosting | ✅ | ❌ |
| Stacking | ✅ | ✅ |
| Blending | ✅ | ✅ |

---

## 3. 四大主流集成策略總覽

| 策略 | 中文名稱 | 概念 | 常見演算法 |
|------|-----------|------|--------------|
| **Bagging** | 裝袋法 | 多模型在隨機取樣的資料上訓練，最後平均或投票 | Random Forest |
| **Boosting** | 提升法 | 逐步修正前一模型的錯誤 | AdaBoost, XGBoost, LightGBM |
| **Stacking** | 堆疊法 | 不同模型的輸出作為次級模型輸入 | sklearn.StackingClassifier |
| **Blending** | 混合法 | 用驗證集混合多模型的輸出 | 自訂 (常見於 Kaggle) |

---

## 4. Titanic 資料介紹與前處理

### 4.1 安裝套件
```bash
pip install pandas numpy seaborn scikit-learn xgboost lightgbm matplotlib
```

### 4.2 載入與清理 Titanic 資料
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 讀取 Titanic 資料
df = pd.read_csv("train.csv")

# 選擇特徵欄位
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

# 數值與類別欄位
numeric_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
categorical_features = ["Sex", "Embarked"]

# 數值與類別欄位前處理
from sklearn.impute import SimpleImputer
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 合併 ColumnTransformer
preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 訓練/測試分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## 5. Bagging：Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```

---

## 6. Boosting：AdaBoost / XGBoost / LightGBM

```python
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb

# AdaBoost
ada_model = Pipeline([
    ("preprocess", preprocess),
    ("model", AdaBoostClassifier(n_estimators=200, learning_rate=0.8, random_state=42))
])

ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))

# XGBoost
xgb_model = Pipeline([
    ("preprocess", preprocess),
    ("model", xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42))
])

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# LightGBM
lgb_model = Pipeline([
    ("preprocess", preprocess),
    ("model", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42))
])

lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
```

---

## 7. Stacking：多模型堆疊學習

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier

base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, random_state=42)),
    ('svc', SVC(probability=True, kernel='rbf', C=2.0, gamma='scale'))
]

stack_model = Pipeline([
    ("preprocess", preprocess),
    ("model", StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(max_iter=1000)))
])

stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))
```

---

## 8. Blending：混合泛化 (Hold-out Blending)

```python
from sklearn.linear_model import LogisticRegression

# 保留部分驗證集作為 blending
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

rf_model.fit(X_tr, y_tr)
xgb_model.fit(X_tr, y_tr)
svc_model = Pipeline([("preprocess", preprocess), ("model", SVC(probability=True))])
svc_model.fit(X_tr, y_tr)

val_rf = rf_model.predict_proba(X_val)[:,1]
val_xgb = xgb_model.predict_proba(X_val)[:,1]
val_svc = svc_model.predict_proba(X_val)[:,1]

blend_X_val = np.vstack([val_rf, val_xgb, val_svc]).T
blend_y_val = y_val.values

blender = LogisticRegression(max_iter=1000)
blender.fit(blend_X_val, blend_y_val)

# 在測試集上做預測
test_rf = rf_model.predict_proba(X_test)[:,1]
test_xgb = xgb_model.predict_proba(X_test)[:,1]
test_svc = svc_model.predict_proba(X_test)[:,1]
blend_X_test = np.vstack([test_rf, test_xgb, test_svc]).T
y_pred_blend = (blender.predict_proba(blend_X_test)[:,1] >= 0.5).astype(int)

print("Blending Accuracy:", accuracy_score(y_test, y_pred_blend))
```

---

## 9. 模型評估與視覺化（Matplotlib + Seaborn）

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc

models = {
    'Random Forest': (y_test, y_pred_rf),
    'XGBoost': (y_test, y_pred_xgb),
    'LightGBM': (y_test, y_pred_lgb),
    'Stacking': (y_test, y_pred_stack),
    'Blending': (y_test, y_pred_blend)
}

# 準確率長條圖
acc = [accuracy_score(y_true, y_pred) for y_true, y_pred in models.values()]
plt.figure(figsize=(8,4))
sns.barplot(x=list(models.keys()), y=acc, palette='viridis')
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=30)
plt.show()

# 混淆矩陣（以 Stacking 為例）
cm = confusion_matrix(y_test, y_pred_stack)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Stacking Confusion Matrix')
plt.show()

# ROC 曲線
plt.figure(figsize=(6,5))
for name, (y_true, y_pred) in models.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{name}')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## 10. AutoML 與深度學習集成延伸

- 可使用 **PyCaret** 或 **AutoGluon** 快速產生最佳集成模型。
- 深度學習集成可混合 CNN / LSTM 輸出結果與樹模型融合。

---

## 11. 總結

> 單一模型 = 一位專家；  
> 集成學習 = 專家委員會。  
> 在 Titanic 這類任務中，集成往往能提升 3~10% 準確率。