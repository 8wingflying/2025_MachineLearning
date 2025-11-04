# 📘 常態分布（Normal Distribution）教學文件

## 🧩 一、常態分布的定義

**常態分布（Normal Distribution）** 是統計學中最重要的連續型機率分布之一，又稱 **高斯分布（Gaussian Distribution）**。  
它的分布圖呈現對等的「鐘形曲線（Bell Curve）」。

常態分布描述了大量自然現象的數值分布，例如：
- 人的身高、體重
- 考試成績
- 測量誤差
- 金融市場報酬率等

---

## 📊 二、常態分布的機率密度函數（PDF）

常態分布的數學式為：

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

其中：
- \( \mu \)：平均數（Mean）
- \( \sigma \)：標準差（Standard Deviation）
- \( \sigma^2 \)：變異數（Variance）

---

## 🧬 三、常態分布的特性

| 特性 | 說明 |
|------|------|
| 對等性 | 以平均數 μ 為中心對等 |
| 平均數、中位數、眷數相等 | μ = median = mode |
| 68-95-99.7 法則 | 約68%的數據落在 μ±1σ；95% 落在 μ±2σ；99.7% 落在 μ±3σ |
| 曲線下的面積總和 = 1 | 代表所有可能結果的總機率 |

---

## 🧮 四、標準常態分布（Standard Normal Distribution）

若：
\[
Z = \frac{X - \mu}{\sigma}
\]
則 Z 服從 **標準常態分布**，其平均數 μ=0，標準差 σ=1。

Z 分數（Z-score）表示資料點距離平均值幾個標準差，用來比較不同常態分布下的數據。

---

## 📘 五、常態分布的應用情境

| 應用場景 | 說明 |
|-----------|------|
| 考試成績分析 | 比較學生表現是否高於平均 |
| 品質控制 | 測量產品尺寸是否符合標準 |
| 金融分析 | 建立投資報酬模型 |
| 統計推論 | 假設檢定與信賴區間估計 |

---

## 🧪 六、Python 實作範例

### 1️⃣ 生成常態分布資料並繪圖
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 設定平均數與標準差
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, mu, sigma)

plt.plot(x, y, color='blue')
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

### 2️⃣ 計算 Z 分數
```python
from scipy.stats import zscore

data = [10, 12, 14, 16, 18]
z_scores = zscore(data)
print(z_scores)
```

### 3️⃣ 計算機率（P(X < a)）
```python
from scipy.stats import norm

prob = norm.cdf(1.96, loc=0, scale=1)  # P(Z < 1.96)
print(f"P(Z < 1.96) = {prob:.4f}")
```

輸出：
```
P(Z < 1.96) = 0.9750
```

---

## 🎯 七、68–95–99.7 規則示意表

| 範圍 | 占比 | 說明 |
|------|------|------|
| μ ± 1σ | 約 68% | 大部分數據集中區 |
| μ ± 2σ | 約 95% | 幾乎所有數據 |
| μ ± 3σ | 約 99.7% | 幾乎不可能之外的數據 |

---

## 📚 八、常見延伸分布

| 分布類型 | 關聯說明 |
|-----------|-----------|
| 標準常態分布 | μ=0, σ=1 的常態分布 |
| 多變量常態分布 | 多維度的常態分布（例如2D, 3D） |
| 對數常態分布 | 資料的對數符合常態分布時使用 |
| 混合常態分布 | 由多個常態分布組合而成（如 GMM） |

---

## 🧩 九、實務範例：考試成績分析

假設某班考試成績服從常態分布：
- 平均數 μ = 70
- 標準差 σ = 10

### 問題：
一位學生考了 85 分，求他高於多少比例的同學？

\[
Z = \frac{85 - 70}{10} = 1.5
\]
查表或使用 `norm.cdf(1.5)`：

```python
from scipy.stats import norm
p = norm.cdf(1.5)
print(f"P(X < 85) = {p:.4f}, 高於比例 = {(1-p)*100:.2f}%")
```

輸出：
```
P(X < 85) = 0.9332, 高於比例 = 6.68%
```

---

## 🧭 十、常態分布與中央極限定理（CLT）

**中央極限定理（Central Limit Theorem）** 指出：  
當樣本數足夠大時，樣本平均數的分布會近似常態分布，即使原本母體不是常態分布。

這使常態分布成為統計推論的基石，廣泛應用於：
- 假設檢定（Hypothesis Testing）
- 信賴區間（Confidence Interval）
- 迴歸分析（Regression Analysis）

---

## ✅ 十一、重點總結

| 主題 | 關鍵概念 |
|------|-----------|
| 定義 | 連續對等分布，平均數 μ、標準差 σ |
| 公式 | \( f(x)=\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \) |
| 特性 | 對等、68-95-99.7 規則 |
| 標準化 | Z = (X−μ)/σ |
| 應用 | 成績、品質控制、金融模型、統計推論 |

---

## 📘 參考資料
- 《Statistics for Data Science Using Python》
- 《Probability and Statistics for Engineers》
- SciPy 官方文件：[https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
- Wikipedia: [

