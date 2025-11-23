# Nonnegative Matrix Factorization
- 2000 [Algorithms for Non-negative Matrix Factorization](https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
- [Lee and Seung (2000)'s Algorithms for Non-negative Matrix Factorization: A Supplementary Proof Guide](https://arxiv.org/abs/2501.11341)
- 2014T [he Why and How of Nonnegative Matrix Factorization](https://arxiv.org/abs/1401.5226)


## 介紹文章
- https://www.geeksforgeeks.org/machine-learning/non-negative-matrix-factorization/
- https://en.wikipedia.org/wiki/Non-negative_matrix_factorization

## 非負矩陣分解（Nonnegative Matrix Factorization, NMF）解法
---

# 1. Multiplicative Update Rules（乘法更新法）
**提出者：** Lee & Seung (2001)

最經典的 NMF 解法，利用梯度下降並配合非負性限制，使更新步驟轉為乘法形式。

### 特色
- 實作最簡單
- 收斂較慢
- 容易卡在局部極值
- 今仍廣泛用於教學與示範

### 常見兩種 Loss
- Frobenius norm
- KL Divergence

---

# 2. Alternating Least Squares (ALS)
交替固定一個矩陣（W 或 H），解另一個矩陣的非負最小平方問題（NNLS）。

### 特色
- 收斂速度比乘法更新快
- 需使用 NNLS 求解器
- 計算較重但效果好

---

# 3. Projected Gradient Descent (PGD) 投影梯度下降
梯度下降後，將 W、H 的負值投影回非負區域。

### 特色
- 可輕鬆加入 L1、L2、稀疏性、平滑性正則化
- 學習率敏感
- 比乘法更新更容易調整與拓展

---

# 4. Coordinate Descent（座標下降法）
逐元素或逐列更新 W、H，使 loss 最小化。  
scikit-learn 的 NMF 預設方法。

### 特色
- 實務中非常穩定且快速
- 內建支援 L1/L2 正則化
- 適用高維資料

---

# 5. HALS / Fast HALS（Hierarchical ALS）
一次更新 W 或 H 的一列（row-wise），比 ALS 更快。

### 特色
- 影像處理常用
- 收斂速度與準確度佳
- 適合高維矩陣

---

# 6. Nesterov Accelerated Gradient（加速梯度）
在 PGD 上加入 momentum。

### 特色
- 收斂速度快於一般梯度法
- 適用於大型資料集
- 用於 Topic Modeling、文件分解

---

# 7. Active Set Method（NNLS Active Set）
把更新視為 NNLS 問題，逐步調整 active set。

### 特色
- 解較精準
- 速度較慢
- 適用精度要求高的情境

---

# 8. Projected Newton / Quasi-Newton Method
利用二階資訊（或近似）加速收斂。

### 特色
- 適合中小規模資料
- 精準且可加入正則化
- 計算成本較高

---

# 9. Sparse NMF（稀疏 NMF）
在 W 或 H 加上 L1 稀疏化。

### 特色
- Topic Modeling 常用：可產生稀疏的主題字詞
- 影像特徵壓縮也常見
- 可自然產生可解釋的 basis

---

# 10. Robust NMF / L1-NMF
將損失函數改為 L1，使模型對離群值較不敏感。

### 特色
- 適用噪聲資料
- 適用 outlier-heavy 資料（如詐欺偵測）

---

# 11. Online NMF（線上 NMF）
利用 mini-batch 更新 W、H，適合大規模或流式資料。

### 特色
- 用於 streaming topic modeling
- 可逐批更新，不需整批資料常駐記憶體
- 高度可擴展

---

# 12. Bayesian NMF（貝葉斯 NMF）
假設因子 W、H 有先驗（如 Gamma），使用：
- Gibbs Sampling
- Variational Inference

### 特色
- 可自動調節 rank
- 有統計上的不確定性度量
- 生醫、來源分離常用

---

# 13. Convex NMF（凸 NMF）
限制 W = X S，使 basis 可被輸入資料本身生成。

### 特色
- 可解釋性佳
- Topic Modeling / 自然語言表示常用
- 對可視化與可解釋 AI 有幫助

---

# 附錄：NMF 求解方法比較表（簡表）

| 方法 | 收斂速度 | 實作難度 | 是否可加入正則化 | 大資料適用性 | 備註 |
|------|-----------|-------------|---------------------|-------------------|------|
| Multiplicative Update | 慢 | 最簡單 | 中等 | 中 | 教學常用 |
| ALS | 中 | 中 | 可 | 中 | 準確度佳 |
| PGD | 中 | 易 | 非常容易 | 中 | 正則化友好 |
| Coordinate Descent | 快 | 中等 | 優 | 中 | sklearn 預設 |
| HALS | 快 | 中等 | 優 | 中 | 高維常用 |
| Nesterov Gradient | 快 | 中等 | 優 | 高 | 適合大型資料 |
| Active Set | 慢 | 難 | 中 | 低 | 精準但慢 |
| (Quasi-)Newton | 中~快 | 中~難 | 可 | 低~中 | 二階方法 |
| Sparse NMF | 中 | 中 | L1 支援 | 中 | Topic/影像 |
| Robust / L1-NMF | 中 | 中 | L1/L2 | 中 | 噪聲資料 |
| Online NMF | 快 | 中等 | 可 | **非常高** | 流式資料 |
| Bayesian NMF | 慢 | 難 | 有先驗 | 中 | 具貝氏意義 |
| Convex NMF | 中 | 中 | 可 | 中 | 可解釋性佳 |

---
