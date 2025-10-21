# 📘 Cross-Validation Visualization Notebook
> 作者：ChatGPT（GPT-5）  
> 說明：以 scikit-learn 示範各種交叉驗證（Cross Validation）方式，並以圖形化方式呈現其資料分割特性。

---

## 🧬 1️⃣ 導入套件與資料生成

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedKFold,
    LeaveOneOut, ShuffleSplit, TimeSeriesSplit
)
from sklearn.datasets import make_classification

# 生成簡單分類資料
X, y = make_classification(
    n_samples=12, n_features=2, n_classes=3,
    n_informative=2, n_redundant=0, random_state=42
)
```

---

## 🥩 2️⃣ 定義通用繪圖函式

```python
def plot_cv_splits(cv, X, y, ax, n_splits, title):
    """
    通用交叉驗證可視化函式
    藍色：訓練集；橘色：驗證集
    """
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        indices = np.arange(len(X))
        ax.scatter(indices[train_idx], [i + 0.5] * len(train_idx),
                   c="skyblue", marker="_", lw=8, label="Train" if i == 0 else "")
        ax.scatter(indices[test_idx], [i + 0.5] * len(test_idx),
                   c="darkorange", marker="_", lw=8, label="Test" if i == 0 else "")

    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(range(1, n_splits + 1))
    ax.set_xlabel("樣本索引")
    ax.set_ylabel("分割次數 (Fold)")
    ax.legend(loc="best")
    ax.set_title(title)
```

---

## 🔹 3️⃣ 各種交叉驗證可視化

### (1) K-Fold Cross Validation

```python
fig, ax = plt.subplots(figsize=(6, 4))
cv = KFold(n_splits=4, shuffle=True, random_state=42)
plot_cv_splits(cv, X, y, ax, 4, "K-Fold Cross Validation")
plt.show()
```

📘 說明：  
- 每橫列為一次折分  
- 橘色 = 驗證集，藍色 = 訓練集  

---

### (2) Stratified K-Fold（分層K折）

```python
fig, ax = plt.subplots(figsize=(6, 4))
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
plot_cv_splits(cv, X, y, ax, 4, "Stratified K-Fold (分層交叉驗證)")
plt.show()
```

📘 說明：  
- 保留每一折的類別比例與整體資料一致  
- 適合類別不均衡的分類任務  

---

### (3) Repeated K-Fold（重複K折）

```python
fig, ax = plt.subplots(figsize=(6, 4))
cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
plot_cv_splits(cv, X, y, ax, 6, "Repeated K-Fold (重複交叉驗證)")
plt.show()
```

📘 說明：  
- 將 K-Fold 多次重複  
- 有助於降低隨機性、提升模型穩定性  

---

### (4) Leave-One-Out（留一法）

```python
fig, ax = plt.subplots(figsize=(6, 4))
cv = LeaveOneOut()
plot_cv_splits(cv, X, y, ax, len(X), "Leave-One-Out Cross Validation")
plt.show()
```

📘 說明：  
- 每次只留一筆資料作為驗證集  
- 適用於資料筆數極少的情境（如醫學研究）  

---

### (5) Shuffle Split（隨機分割）

```python
fig, ax = plt.subplots(figsize=(6, 4))
cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=42)
plot_cv_splits(cv, X, y, ax, 4, "Shuffle Split (隨機分割驗證)")
plt.show()
```

📘 說明：  
- 每次隨機抽樣不同的訓練 / 驗證資料  
- 適合大型資料集，測試泛化能力  

---

### (6) Time Series Split（時間序列驗證）

```python
fig, ax = plt.subplots(figsize=(6, 4))
cv = TimeSeriesSplit(n_splits=4)
plot_cv_splits(cv, X, y, ax, 4, "Time Series Split (時間序列交叉驗證)")
plt.show()
```

🕰️ 說明：  
- 不打亂時間順序  
- 模擬未來資料預測  
- 適合時間序列分析（股價、氣象、IoT）  

---

## 📊 4️⃣ 方法比較總表

| 方法 | 是否打亂資料 | 保留類別比例 | 適合任務 | 特點 |
|------|---------------|----------------|------------|------------|
| **KFold** | ✅ 可選 | ❌ | 分類 / 回歸 | 最常用、穩定 |
| **StratifiedKFold** | ✅ | ✅ | 分類 | 保持類別比例一致 |
| **RepeatedKFold** | ✅ | ❌ | 小型資料 | 降低隨機偏差 |
| **LeaveOneOut** | ❌ | ❌ | 小樣本 | 精準但耗時 |
| **ShuffleSplit** | ✅ | 可選 | 大型資料 | 高彈性 |
| **TimeSeriesSplit** | ❌ | ❌ | 時間序列 | 保留時間順序 |

---

## 🧠 5️⃣ 結論與教學建議

📘 建議選擇方式：
- 一般任務 → `KFold`
- 類別不均衡 → `StratifiedKFold`
- 小資料集 → `LeaveOneOut`
- 大資料 + 隨機抽樣 → `ShuffleSplit`
- 時間序列 → `TimeSeriesSplit`

💡 **延伸應用：**
- 可與 `GridSearchCV`、`RandomizedSearchCV` 搭配使用以調整超參數  
- 可結合 `matplotlib.animation` 動態展示資料折分過程  
- 可擴充為 AutoML pipeline 的一部分  

---

# ✅ 結束
> 本筆記可直接複製至 Jupyter Notebook 或 VSCode Markdown Preview 內執行。
> 執行後會自動顯示每種交叉驗證的視覺化圖。

