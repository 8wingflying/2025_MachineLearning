# 📘 高斯分布的最大概似估計（MLE for Gaussian Distribution）

## 🧩 一、前言

在統計學與機器學習中，我們常使用最大概似估計（Maximum Likelihood Estimation, MLE）來估計機率分布的參數。  
對於高斯分布（常態分布），MLE 提供了一種數學上最自然的方式估計其平均數（μ）與變異數（σ²）。

---

## 📊 二、常態分布定義

常態分布的機率密度函數（PDF）為：

\[
f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(x - \mu)^2}{2\sigma^2}\right]
\]

參數：
- μ：期望值（mean）
- σ²：變異數（variance）

---

## 🥮 三、似然函數 (Likelihood Function)

假設有樣本集  
\[
X = \{x_1, x_2, ..., x_n\}
\]  
每個樣本獨立且服從相同的高斯分布：

\[
L(\mu, \sigma^2) = \prod_{i=1}^n f(x_i|\mu, \sigma^2)
\]

通常取對數以便計算：

\[
\ln L(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
\]

---

## 🔢 四、對參數求偏微分

### (1) 對 μ 求偏微分：

\[
\frac{\partial \ln L}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu)
\]

令其為 0：

\[
\sum_{i=1}^n (x_i - \mu) = 0 \Rightarrow \hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i
\]

✅ **結果：MLE 對 μ 的估計值就是樣本平均數。**

---

### (2) 對 σ² 求偏微分：

\[
\frac{\partial \ln L}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2
\]

令其為 0：

\[
-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2 = 0
\]

\[
\Rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2
\]

✅ **結果：MLE 對 σ² 的估計值是樣本方差（但母體分母為 n，而非 n−1）。**

---

## 🧠 五、與樣本方差的關係

| 估計方法 | 變異數公式 | 分母 | 特性 |
|-----------|-------------|------|------|
| MLE | \( \frac{1}{n}\sum(x_i - \bar{x})^2 \) | n | 無偏但低估母體變異 |
| 無偏估計 | \( \frac{1}{n-1}\sum(x_i - \bar{x})^2 \) | n−1 | 常用於統計推論 |

---

## 💻 六、Python 實作

```python
import numpy as np

# 假設我們有樣本資料
data = np.array([4.0, 5.2, 6.1, 5.8, 4.9])

# MLE 估計 μ 與 σ²
mu_mle = np.mean(data)
sigma2_mle = np.mean((data - mu_mle)**2)

print(f"MLE μ = {mu_mle:.4f}")
print(f"MLE σ² = {sigma2_mle:.4f}")
```

---

## 📈 七、圖示說明（直覺理解）

![Gaussian Curve Illustration](./images/gaussian_mle_diagram.png)

- **μ (平均值)** 決定了曲線的中心位置。
- **σ (標準差)** 決定了曲線的寬度（越大越平緩）。
- MLE 嘗試找到一組參數，使得「觀察到的資料出現的機率最大。」

---

## 🧩 八、總結

| 項目 | MLE 結果 | 解釋 |
|------|------------|------|
| 平均值 μ | \( \hat{\mu} = \frac{1}{n}\sum x_i \) | 樣本平均數 |
| 變異數 σ² | \( \hat{\sigma}^2 = \frac{1}{n}\sum (x_i - \hat{\mu})^2 \) | 樣本方差 (n分母) |
| 注意 | 無偏估計需用 n−1 | 統計學常見修正 |

---

📘 **延伸閱讀**
- Bishop, *Pattern Recognition and Machine Learning* (2006)
- Murphy, *Machine Learning: A Probabilistic Perspective* (2012)
- Numpy 官方文件：[https://numpy.org/doc/](https://numpy.org/doc/)

