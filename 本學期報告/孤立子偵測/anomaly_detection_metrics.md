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

## 📈 三、Python 範例（以 Isolation Forest 為例）

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.datasets import make_blobs

# 生成示例資料
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
rng = np.random.RandomState(42)
X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))
X_total = np.vstack([X, X_outliers])
y_true = np.array([0] * 300 + [1] * 20)  # 0=正常, 1=異常

# 模型訓練
clf = IsolationForest(contamination=0.06, random_state=42)
y_pred = clf.fit_predict(X_total)
y_pred = np.where(y_pred == -1, 1, 0)  # 轉成 0/1 標記

# 評估
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")
print(f"ROC-AUC:   {auc:.2f}")
```

---

## 📊 四、視覺化：ROC 與 PR 曲線

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

fpr, tpr, _ = roc_curve(y_true, y_pred)
precision, recall, _ = precision_recall_curve(y_true, y_pred)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, color='blue')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.subplot(1,2,2)
plt.plot(recall, precision, color='green')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()
```

---

## 🧠 五、實務建議

1. **極度不平衡資料集** → 優先觀察 **Precision、Recall、F1、PR-AUC**。  
2. **缺乏標籤** → 可採「模型內部分數（如重建誤差）」＋「人工驗證樣本」混合評估。  
3. **多模型比較** → 對相同資料集，統一使用 ROC–AUC 或 F1-score 比較性能。  
4. **應用場景取向** → 需根據誤報／漏報代價（FP/FN Cost）選擇合適閾值。

---

📘 **作者建議**：將此文件保存為 `anomaly_detection_metrics.md`（UTF-8 編碼），方便課程或實驗報告引用。