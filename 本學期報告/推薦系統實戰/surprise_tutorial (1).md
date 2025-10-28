# 🎯 Surprise 套件教學文件  
> Python 推薦系統套件 — Scikit-Surprise 完整教學  

---

## 📘 一、Surprise 套件簡介  

**Surprise（Simple Python RecommendatIon System Engine）** 是一個用於建立與評估推薦系統的 Python 套件，專門處理 **協同過濾（Collaborative Filtering）** 問題。  
它可幫助你快速：  
- 載入常見資料集（如 MovieLens）  
- 建立不同的推薦模型  
- 進行交叉驗證與評估  
- 自訂演算法（KNN、SVD、Baseline 等）  

### 🔧 安裝  
```bash
pip install scikit-surprise
```

---

## 📚 二、主要模組與類別概覽  

| 模組 / 類別 | 功能 |
|--------------|------|
| `Dataset` | 資料集載入與建立 |
| `Reader` | 定義資料格式 |
| `Trainset` | Surprise 的內部訓練資料結構 |
| `SVD`, `KNNBasic`, `KNNWithMeans`, `NMF` | 主要演算法 |
| `cross_validate`, `train_test_split` | 評估工具 |
| `accuracy` | 評估 RMSE、MAE 指標 |

---

## 🧠 三、推薦系統基本原理  

Surprise 支援兩大類推薦方法：

### 1️⃣ 協同過濾（Collaborative Filtering）
根據使用者歷史行為找出相似性。  
- **User-based CF**：找出與你相似的使用者。  
- **Item-based CF**：找出與你喜歡的項目相似的其他項目。  

### 2️⃣ 矩陣分解（Matrix Factorization）
透過 **SVD / NMF** 將使用者-項目矩陣分解為潛在向量，用於預測未知評分。

---

## 🧩 四、資料集載入與前處理  

### (1) 使用內建資料集（MovieLens）  
```python
from surprise import Dataset
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)
```

### (2) 載入自訂 CSV 資料  
```python
from surprise import Dataset, Reader

reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)
```

---

## ⚙️ 五、主要演算法與範例  

### (1) SVD（矩陣分解）
```python
from surprise import SVD, Dataset
from surprise.model_selection import cross_validate

algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### (2) KNN 基礎模型  
```python
from surprise import KNNBasic

sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
```

### (3) BaselineOnly 模型（含偏移調整）
```python
from surprise import BaselineOnly

algo = BaselineOnly()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
```

---

## 🧪 六、模型訓練與預測  

```python
from surprise import SVD, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy

trainset, testset = train_test_split(data, test_size=0.25)
algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)
accuracy.rmse(predictions)
```

---

## 📈 七、推薦結果輸出  

```python
from surprise import Dataset, Reader, SVD

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

# 為使用者 196 預測對物品 302 的評分
pred = algo.predict(uid=196, iid=302)
print(pred)
```

輸出格式：
```
Prediction(uid=196, iid=302, r_ui=None, est=4.25, details={'was_impossible': False})
```

---

## 📊 八、模型評估與可視化  

```python
import matplotlib.pyplot as plt
from surprise.model_selection import cross_validate
import pandas as pd

results = cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
df = pd.DataFrame(results)

plt.figure(figsize=(6,4))
plt.plot(df['test_rmse'], label='Test RMSE')
plt.plot(df['train_rmse'], label='Train RMSE')
plt.legend()
plt.title('SVD Model RMSE Comparison')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.show()
```

---

## 🧮 九、KNN 類型比較  

| 模型名稱 | 說明 |
|-----------|------|
| `KNNBasic` | 最簡單的 KNN，僅根據相似度加權 |
| `KNNWithMeans` | 平均中心化（mean-centered）調整 |
| `KNNWithZScore` | 標準化 Z-Score 調整 |
| `KNNBaseline` | 加入 Baseline 偏差校正 |

---

## 🧱 十、自訂推薦系統流程範例  

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 載入自訂資料
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)

trainset, testset = train_test_split(data, test_size=0.2)

algo = SVD(n_factors=100, n_epochs=20, reg_all=0.02)
algo.fit(trainset)
predictions = algo.test(testset)

print("RMSE:", accuracy.rmse(predictions))
```

---

## 💡 十一、Top-N 推薦系統範例（完整 Python 程式）  

```python
from surprise import SVD, Dataset
from surprise.model_selection import train_test_split
import pandas as pd

# 載入內建資料集
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# 訓練模型
algo = SVD()
algo.fit(trainset)

# 建立推薦函數
def get_top_n(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        top_n.setdefault(uid, []).append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# 測試與生成 Top-N 推薦
predictions = algo.test(testset)
top_n = get_top_n(predictions, n=5)

# 顯示部分使用者的推薦清單
for uid, user_ratings in list(top_n.items())[:3]:
    print(f"使用者 {uid} 的推薦清單:")
    for (iid, rating) in user_ratings:
        print(f"\t物品 {iid}: 預測評分 {rating:.2f}")
```

📊 此範例將每位使用者的推薦結果儲存在字典 `top_n` 中，可進一步整合至前端（例如 **Streamlit** 或 **Flask**）進行互動展示。

---

## 🧠 十二、常用評估指標  

| 指標 | 定義 | 用途 |
|------|------|------|
| **RMSE** | Root Mean Squared Error | 衡量預測誤差的平均平方根 |
| **MAE** | Mean Absolute Error | 衡量平均絕對誤差 |
| **Precision / Recall / F1** | 在 Top-N 推薦情境下使用 | 評估推薦品質 |

---

## 🚀 十三、延伸應用  

- **Top-N 推薦清單**  
  根據預測結果排序，挑出使用者最可能喜歡的項目。  
- **混合式模型（Hybrid Recommender）**  
  結合內容過濾與協同過濾。  
- **與 Pandas / Streamlit 整合**  
  可視化推薦結果、建立互動式推薦系統 Dashboard。

---

## 📦 十四、參考資料  
- Surprise 官方文件：[https://surpriselib.com](https://surpriselib.com)  
- MovieLens Dataset：[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)  
- Paper: Koren, Bell, Volinsky (2009). *Matrix Factorization Techniques for Recommender Systems.*

