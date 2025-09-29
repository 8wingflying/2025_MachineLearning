## 不平衡學習(Imbalance Learning)

## 參考資料
- BalanceBenchmark: A Survey for Multimodal Imbalance Learning
- https://arxiv.org/abs/2502.10816
- A review of methods for imbalanced multi-label classification
- https://www.sciencedirect.com/science/article/abs/pii/S0031320321001527
- https://blog.csdn.net/hren_ron/article/details/81172044
- A survey on imbalanced learning: latest research, applications and future directions
- https://link.springer.com/article/10.1007/s10462-024-10759-6



#### Python套件
- imbalanced-learn
- https://imbalanced-learn.org/stable/
- pip install imbalanced-learn
- https://github.com/ycz3792/imbalance-learn

#### 不平衡學習的 Python 範例，包含三個部分：
- 原始資料集 (不平衡)
- SMOTE (過取樣)
- 加權分類器 (Logistic Regression with class_weight)
- 模型評估 (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# 1️⃣ 建立不平衡資料集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, weights=[0.95, 0.05], random_state=42)

print("原始類別分布:", np.bincount(y))  # [950, 50]

# 分割訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2️⃣ 使用 SMOTE 做過取樣
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("SMOTE 後類別分布:", np.bincount(y_res))  # [665, 665]

# 3️⃣ 訓練加權分類器
clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
clf.fit(X_res, y_res)

# 4️⃣ 預測與評估
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n分類報告 (Precision, Recall, F1):")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

print(f"ROC-AUC = {roc_auc:.3f}")
print(f"PR-AUC = {pr_auc:.3f}")

# 📊 視覺化 Precision-Recall 曲線
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker=".", label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
```
