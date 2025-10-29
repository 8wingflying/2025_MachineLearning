# Seaborn 統計圖分類總覽（Statistical Plots Overview）

> 作者：T Ben  
> 語言：繁體中文  
> 檔名：`seaborn_統計圖分類總覽.md`

---

## 🧮 Seaborn 統計圖分類與用途一覽表

Seaborn 的核心價值在於 **統計型視覺化**，能夠自動計算平均值、信賴區間、迴歸線等統計資訊。以下表格整理出 Seaborn 主要統計圖的類別、用途與對應函式。

---

### 📊 1️⃣ 分布型圖（Distribution Plots）

| 圖表類型 | 函式名稱 | 功能說明 | 常用參數 | 範例用途 |
|-----------|-----------|-----------|------------|-----------|
| 直方圖 (Histogram) | `sns.histplot()` | 顯示變量分布頻率 | `bins`, `kde`, `hue` | 數值分布、年齡分布 |
| KDE 密度圖 | `sns.kdeplot()` | 以核密度估計顯示分布曲線 | `bw_adjust`, `fill` | 平滑分布曲線 |
| 離散計數圖 | `sns.countplot()` | 顯示類別出現次數 | `hue`, `order` | 類別資料統計 |
| ECDF 圖 | `sns.ecdfplot()` | 顯示累積分布函數 | `complementary` | 累積百分比分析 |

🧩 **示例：**
```python
sns.histplot(data=tips, x="total_bill", kde=True, hue="sex")
plt.title("消費金額分布（含 KDE 曲線）")
plt.show()
```

---

### 📈 2️⃣ 關聯型圖（Relational Plots）

| 圖表類型 | 函式名稱 | 功能說明 | 常用參數 | 範例用途 |
|-----------|-----------|-----------|------------|-----------|
| 散點圖 | `sns.scatterplot()` | 顯示兩變量間的關係 | `hue`, `size`, `style` | 消費 vs 小費 |
| 折線圖 | `sns.lineplot()` | 顯示隨時間或數值的變化趨勢 | `estimator`, `ci`, `hue` | 趨勢分析 |

🧩 **示例：**
```python
sns.lineplot(data=tips, x="size", y="tip", hue="day")
plt.title("不同日期下用餐人數與小費關係")
plt.show()
```

---

### 📦 3️⃣ 分類型圖（Categorical Plots）

| 圖表類型 | 函式名稱 | 功能說明 | 常用參數 | 範例用途 |
|-----------|-----------|-----------|------------|-----------|
| 箱型圖 | `sns.boxplot()` | 顯示中位數、四分位與離群值 | `hue`, `orient` | 不同性別的消費分布 |
| 小提琴圖 | `sns.violinplot()` | 結合箱型與密度估計 | `split`, `inner` | 類別分布比較 |
| 長條圖 | `sns.barplot()` | 顯示平均值與信賴區間 | `ci`, `estimator` | 每週平均消費 |
| 點圖 | `sns.pointplot()` | 顯示均值隨類別變化 | `join`, `markers` | 多組比較 |
| 蜂群圖 | `sns.swarmplot()` | 類別資料散點顯示 | `hue`, `size` | 資料點分布可視化 |
| 箱型 + 蜂群 | `sns.boxenplot()` | 更平滑的箱型圖（適用大樣本） | `scale`, `outlier_prop` | 大樣本資料分析 |

🧩 **示例：**
```python
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
plt.title("每週消費金額箱型圖（依性別分色）")
plt.show()
```

---

### 🔬 4️⃣ 迴歸型圖（Regression Plots）

| 圖表類型 | 函式名稱 | 功能說明 | 常用參數 | 範例用途 |
|-----------|-----------|-----------|------------|-----------|
| 基本迴歸圖 | `sns.regplot()` | 顯示線性迴歸線與信賴區間 | `order`, `ci`, `line_kws` | 小費與帳單金額關係 |
| 自動分面迴歸圖 | `sns.lmplot()` | 可分組、多圖比較的迴歸繪圖 | `col`, `row`, `hue` | 各天迴歸比較 |

🧩 **示例：**
```python
sns.lmplot(data=tips, x="total_bill", y="tip", hue="day", aspect=1.2)
plt.title("各日期消費 vs 小費線性回歸圖")
plt.show()
```

---

### 🧠 5️⃣ 矩陣型圖（Matrix Plots）

| 圖表類型 | 函式名稱 | 功能說明 | 常用參數 | 範例用途 |
|-----------|-----------|-----------|------------|-----------|
| 熱圖 | `sns.heatmap()` | 顯示矩陣或相關係數關係 | `annot`, `cmap`, `center` | 變數相關性分析 |
| 聚類熱圖 | `sns.clustermap()` | 熱圖 + 階層分群 | `method`, `metric` | 相似變數群組化 |

🧩 **示例：**
```python
corr = tips.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("相關性熱圖")
plt.show()
```

---

### 🧩 6️⃣ 配對型圖（Pairwise Plots）

| 圖表類型 | 函式名稱 | 功能說明 | 常用參數 | 範例用途 |
|-----------|-----------|-----------|------------|-----------|
| 配對圖 | `sns.pairplot()` | 顯示多變量間的成對關係 | `hue`, `diag_kind` | 多維資料探索 |
| 聯合圖 | `sns.jointplot()` | 結合散點與分布圖 | `kind`, `hue` | 雙變量分析 |

🧩 **示例：**
```python
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex", color="purple")
plt.show()
```

---

### 🕒 7️⃣ 時間序列圖（Time Series Plots）

| 圖表類型 | 函式名稱 | 功能說明 | 常用參數 | 範例用途 |
|-----------|-----------|-----------|------------|-----------|
| 折線時間圖 | `sns.lineplot()` | 支援時間序列平均與區間顯示 | `ci`, `estimator` | 銷售趨勢分析 |
| Smooth 曲線圖 | `sns.relplot(kind='line')` | 多維時間序列 | `col`, `row`, `hue` | 分組趨勢可視化 |

---

### 🧩 8️⃣ 分群與網格型圖（Grid & Multiplot）

| 圖表類型 | 函式名稱 | 功能說明 | 範例用途 |
|-----------|-----------|-----------|-----------|
| FacetGrid | `sns.FacetGrid()` | 多面板繪圖容器 | 依類別自動分組繪圖 |
| PairGrid | `sns.PairGrid()` | 自訂配對圖矩陣 | 客製化多變量關聯矩陣 |
| JointGrid | `sns.JointGrid()` | 自訂雙變量分析圖 | 結合散點、邊緣分布與 KDE |

---

📘 **總結：**
- 若要分析分布 ➜ `histplot` / `kdeplot`  
- 若要看變量關係 ➜ `scatterplot` / `lineplot`  
- 若要分析類別 ➜ `boxplot` / `violinplot` / `barplot`  
- 若要分析關聯性 ➜ `heatmap` / `pairplot`  
- 若要顯示迴歸趨勢 ➜ `lmplot` / `regplot`

---

> 💡 小提示：
> 你可以用 `sns.set_theme(style='whitegrid', palette='deep')` 統一設定所有圖的風格與配色。

