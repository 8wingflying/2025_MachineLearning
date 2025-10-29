# Seaborn 教學文件（含完整 Python 範例與圖表）

> 作者：T Ben  
> 語言：繁體中文  
> 編碼：UTF-8  
> 檔名：`seaborn_教學文件.md`

---

## 🧭 章節目錄

1. [Seaborn 概述](#1)
2. [安裝與匯入](#2)
3. [Seaborn 與 Matplotlib 的關係](#3)
4. [資料集與基本繪圖範例](#4)
5. [統計繪圖函式總覽](#5)
6. [樣式與調色盤設定](#6)
7. [FacetGrid 與多變量視覺化](#7)
8. [回歸與分類分析繪圖](#8)
9. [熱圖與相關性分析](#9)
10. [綜合範例：Iris 資料集可視化](#10)
11. [延伸章節：Seaborn vs Matplotlib 比較分析](#11)
12. [結語與參考資源](#12)

---

<a id="1"></a>
## 1️⃣ Seaborn 概述

Seaborn 是基於 **Matplotlib** 的高階統計繪圖套件，專為簡化資料探索與分析設計。  
其優點包括：
- 支援 **DataFrame 直接操作**
- 內建多種美觀樣式與配色
- 能快速產生統計圖（如箱型圖、熱圖、回歸圖）
- 與 Pandas 無縫整合

---

<a id="2"></a>
## 2️⃣ 安裝與匯入

```python
# 安裝 Seaborn
!pip install seaborn matplotlib pandas

# 匯入模組
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

---

<a id="3"></a>
## 3️⃣ Seaborn 與 Matplotlib 的關係

| 比較項目 | Matplotlib | Seaborn |
|-----------|-------------|----------|
| 定位 | 基礎繪圖庫 | 高階統計繪圖 |
| 操作資料 | 陣列為主 | DataFrame 為主 |
| 美觀度 | 須自行調整 | 內建主題與調色 |
| 用途 | 自訂圖表 | 快速分析資料 |

```python
# Matplotlib 範例
plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.title("Matplotlib 範例")
plt.show()

# Seaborn 範例
sns.lineplot(x=[1, 2, 3, 4], y=[10, 20, 25, 30])
plt.title("Seaborn 範例")
plt.show()
```

![Lineplot Example](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/_images/seaborn_lineplot_example.png)

---

<a id="4"></a>
## 4️⃣ 資料集與基本繪圖範例

Seaborn 內建多個樣例資料集，例如：`tips`, `iris`, `penguins`

```python
# 載入資料集
tips = sns.load_dataset("tips")

# 顯示前 5 筆資料
print(tips.head())

# 繪製散點圖
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.title("消費金額 vs 小費（依日期分色）")
plt.show()
```

![Scatterplot Example](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/_images/seaborn_scatterplot_example.png)

---

<a id="5"></a>
## 5️⃣ 統計繪圖函式總覽

| 圖表類型 | 函式名稱 | 用途 |
|-----------|------------|------|
| 散點圖 | `sns.scatterplot()` | 顯示變量關係 |
| 折線圖 | `sns.lineplot()` | 顯示趨勢 |
| 箱型圖 | `sns.boxplot()` | 分布與離群值 |
| 小提琴圖 | `sns.violinplot()` | 分布密度與中位數 |
| 長條圖 | `sns.barplot()` | 類別平均值 |
| 熱圖 | `sns.heatmap()` | 顯示矩陣關係 |
| 配對圖 | `sns.pairplot()` | 全變量關聯矩陣 |

---

<a id="6"></a>
## 6️⃣ 樣式與調色盤設定

```python
# 設定樣式
sns.set_style("whitegrid")

# 調色盤預覽
sns.palplot(sns.color_palette("pastel"))

# 套用主題樣式
sns.set_theme(style="darkgrid", palette="muted")
```

可選樣式：
- `"darkgrid"`
- `"whitegrid"`
- `"dark"`
- `"white"`
- `"ticks"`

---

<a id="7"></a>
## 7️⃣ FacetGrid 與多變量視覺化

```python
# 以性別與星期分組顯示小費散點圖
g = sns.FacetGrid(tips, col="sex", row="day", hue="smoker")
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
g.add_legend()
plt.show()
```

---

<a id="8"></a>
## 8️⃣ 回歸與分類分析繪圖

```python
# 簡單線性回歸圖
sns.lmplot(data=tips, x="total_bill", y="tip", height=5, aspect=1.2)
plt.title("線性回歸分析")
plt.show()

# 類別長條圖
sns.barplot(data=tips, x="day", y="total_bill", hue="sex", ci="sd")
plt.title("平均消費金額（依性別與星期）")
plt.show()
```

---

<a id="9"></a>
## 9️⃣ 熱圖與相關性分析

```python
# 計算相關係數矩陣
corr = tips.corr(numeric_only=True)

# 繪製熱圖
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("相關性熱圖")
plt.show()
```

![Heatmap Example](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/_images/seaborn_heatmap_example.png)

---

<a id="10"></a>
## 🔟 綜合範例：Iris 資料集可視化

```python
# 載入 Iris 資料集
iris = sns.load_dataset("iris")

# 多變量分佈配對圖
sns.pairplot(iris, hue="species", diag_kind="kde", palette="husl")
plt.suptitle("Iris 資料集多維度視覺化", y=1.02)
plt.show()

# 箱型圖
sns.boxplot(data=iris, x="species", y="sepal_length", palette="Set2")
plt.title("花萼長度分佈（依花種）")
plt.show()
```

---

<a id="11"></a>
## 🔬 延伸章節：Seaborn vs Matplotlib 比較分析 × Python 實作

| 項目 | Seaborn | Matplotlib |
|------|----------|-------------|
| 操作資料結構 | DataFrame 為主 | 陣列為主 |
| 美觀預設 | 自動 | 手動設定 |
| 圖表風格 | 現代化統計風格 | 傳統繪圖風格 |
| 顏色控制 | palette 參數 | color 參數 |
| 最適用場合 | 資料探索與分析 | 高度客製化繪圖 |

### 🎨 實作對照：

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Matplotlib
plt.figure(figsize=(8,4))
plt.plot(x, y, color='red', linestyle='--')
plt.title('Matplotlib 繪圖')
plt.show()

# Seaborn
sns.set_theme(style="whitegrid")
sns.lineplot(x=x, y=y, color='blue')
plt.title('Seaborn 繪圖')
plt.show()
```

![Seaborn vs Matplotlib](https://seaborn.pydata.org/_images/function_overview_8_0.png)

---

<a id="12"></a>
## 🧩 結語與參考資源

Seaborn 是 Python 中最直覺且強大的資料視覺化工具之一。  
結合 Pandas、Matplotlib，可快速實現美觀且具分析性的圖表。

📘 **推薦參考：**
- 官方文件：[https://seaborn.pydata.org](https://seaborn.pydata.org)
- 教學範例：[Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- 搭配 Pandas 資料視覺化教學

---

> 📦 作者附註：可整合進 `data_viz_tutorials.md` 或作為 `Chapter 5：Seaborn 高階繪圖教學`

