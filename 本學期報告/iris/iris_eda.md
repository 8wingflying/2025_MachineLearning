# 🌸 Iris 資料集探索性資料分析（EDA）教學  
**Exploratory Data Analysis (EDA) of the Iris Dataset**

---

## 📘 一、資料集介紹 (Dataset Overview)

Iris 資料集是統計學與機器學習中最著名的範例之一，由 Ronald A. Fisher 於 1936 年提出。  
它包含 150 筆鳴尾花樣本，共分為三個品種：

- **Setosa**
- **Versicolor**
- **Virginica**

### 特徵說明 (Features)

| 特徵名稱 | 英文欄位 | 單位 | 說明 |
|-----------|-----------|------|------|
| 萃片長度 | sepal length | cm | 花萃的長度 |
| 萃片寬度 | sepal width | cm | 花萃的寬度 |
| 花瓣長度 | petal length | cm | 花瓣的長度 |
| 花瓣寬度 | petal width | cm | 花瓣的寬度 |

---

## 🧩 二、載入資料 (Load Dataset)

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 載入內建的 Iris 資料集
iris = sns.load_dataset("iris")

# 查看前五筆資料
print(iris.head())
```

---

## 📊 三、基本資料檢查 (Basic Info & Summary)

```python
# 資料基本資訊
print(iris.info())

# 統計摘要
print(iris.describe())

# 檢查缺失值
print(iris.isnull().sum())
```

---

## 📈 四、單變量分析 (Univariate Analysis)

### 1️⃣ 各特徵分佈（直方圖 Histogram）
```python
iris.hist(figsize=(10, 8), bins=20)
plt.suptitle("Iris 特徵分佈直方圖", fontsize=14)
plt.show()
```

### 2️⃣ 使用 Seaborn Pairplot 顯示特徵關係
```python
sns.pairplot(iris, hue="species", diag_kind="kde")
plt.suptitle("Iris 資料集 Pairplot", y=1.02)
plt.show()
```

📘 **說明：**  
- 對角線顯示各特徵的機率密度（KDE）。  
- 其他格子為兩特徵之間的散點圖。  
- 顏色區分三個品種。

---

## 🧮 五、多變量分析 (Multivariate Analysis)

### 1️⃣ 各特徵與品種之箱型圖 (Boxplot)
```python
plt.figure(figsize=(10, 6))
sns.boxplot(x="species", y="sepal_length", data=iris)
plt.title("不同品種之萃片長度分佈")
plt.show()

# 可重複用於所有特徵
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for f in features:
    sns.boxplot(x="species", y=f, data=iris)
    plt.title(f"不同品種之 {f} 分佈")
    plt.show()
```

### 2️⃣ 特徵間的相關熱圖 (Correlation Heatmap)
```python
plt.figure(figsize=(8, 6))
sns.heatmap(iris.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特徵間相關性矩陣 (Correlation Matrix)")
plt.show()
```

🔍 **觀察：**
- 花瓣長度與花瓣寬度高度正相關。
- 萃片長度與花瓣長度也有明顯關聯。
- 萃片寬度的相關性較低。

---

## 🔬 六、降維與群集趨勢 (PCA Visualization)

### 使用 PCA 將特徵降至 2 維並視覺化
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
components = pca.fit_transform(iris.drop(columns="species"))

iris_pca = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
iris_pca['species'] = iris['species']

sns.scatterplot(x='PCA1', y='PCA2', hue='species', data=iris_pca, s=80)
plt.title("Iris PCA 降維後視覺化")
plt.show()
```

---

## 📚 七、觀察與結論 (Insights & Conclusions)

| 分析面向 | 發現重點 |
|-----------|-----------|
| 花瓣長度與花瓣寬度 | 具有高度正相關，是分類的關鍵特徵 |
| Setosa | 特徵分佈明顯獨立，易於分類 |
| Versicolor vs Virginica | 有部分重疊，需進一步模型區分 |
| PCA 結果 | 兩個主成分已能顯著區分三種花 |

---

## 🚀 八、延伸分析方向 (Next Steps)

1. 使用 **K-Means** 進行無監督分群。
2. 構建 **分類模型（SVM / Logistic Regression）** 進行分類預測。
3. 分析 **特徵重要性（Feature Importance）**，探討各特徵的貢獻度。

---

## 📁 附錄：完整 Python 套件需求 (Dependencies)

```bash
pip install pandas seaborn matplotlib scikit-learn
```

---

🗕 **建立日期：** 2025-10-28  
✍️ **作者：** ChatGPT 教學助手  
🧠 **主題：** Exploratory Data Analysis (EDA) on Iris Dataset  

