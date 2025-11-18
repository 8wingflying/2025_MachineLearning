##

```python

# -------------------------------------------------------------
# 使用 SVD 分析信用卡詐欺 Fraud Detection
# 內容包含：
# 1. 讀取資料
# 2. 標準化資料
# 3. SVD 矩陣分解
# 4. 2D 視覺化 (SVD1 vs SVD2)
# 5. Reconstruction Error 作為異常分數
# -------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 1. 載入資料
# -------------------------------------------------------------
df = pd.read_csv("creditcard.csv")

print("資料筆數：", df.shape)
print(df.head())

X = df.drop(columns=["Class"])
y = df["Class"]

# -------------------------------------------------------------
# 2. 資料標準化
# -------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("標準化後資料形狀：", X_scaled.shape)

# -------------------------------------------------------------
# 3. 套用 SVD 分解
# X ≈ U Σ V^T
# -------------------------------------------------------------
U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)

print("U shape:", U.shape)
print("S shape:", S.shape)
print("VT shape:", VT.shape)

# -------------------------------------------------------------
# 4. 顯示奇異值解釋變異（類似 PCA 的 explained variance）
# -------------------------------------------------------------
explained_variance = (S**2) / np.sum(S**2)
cumulative = np.cumsum(explained_variance)

plt.figure(figsize=(10,5))
plt.plot(explained_variance[:20], marker="o")
plt.title("前 20 個奇異值的解釋變異")
plt.xlabel("成分 index")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# 5. 建立 SVD 2D 降維視覺化
#    SVD1 = U[:,0] * S[0]
#    SVD2 = U[:,1] * S[1]
# -------------------------------------------------------------
X_svd_2d = U[:, :2] * S[:2]

svd_df = pd.DataFrame({
    "SVD1": X_svd_2d[:, 0],
    "SVD2": X_svd_2d[:, 1],
    "Class": y
})

plt.figure(figsize=(10, 7))
plt.scatter(
    svd_df[svd_df["Class"] == 0]["SVD1"],
    svd_df[svd_df["Class"] == 0]["SVD2"],
    s=2, alpha=0.3, label="Normal"
)
plt.scatter(
    svd_df[svd_df["Class"] == 1]["SVD1"],
    svd_df[svd_df["Class"] == 1]["SVD2"],
    s=20, color="red", alpha=0.7, label="Fraud"
)
plt.title("SVD 降維後之 2D 視覺化 (信用卡詐欺資料)")
plt.xlabel("SVD1")
plt.ylabel("SVD2")
plt.legend()
plt.show()

# -------------------------------------------------------------
# 6. SVD Reconstruction Error 異常偵測
#    只保留前 k 個奇異值（低秩重建）
# -------------------------------------------------------------
k = 5  # 可依需求調整保留幾個主要成分

U_k = U[:, :k]
S_k = np.diag(S[:k])
V_k = VT[:k, :]

# X_recon = U Σ V^T
X_recon = np.dot(np.dot(U_k, S_k), V_k)

# Reconstruction Error（每筆資料）
recon_error = np.mean((X_scaled - X_recon) ** 2, axis=1)
df["recon_error"] = recon_error

# -------------------------------------------------------------
# 7. 顯示正常 vs 詐欺的 Reconstruction Error
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df["Class"] == 0]["recon_error"], label="Normal")
sns.kdeplot(df[df["Class"] == 1]["recon_error"], label="Fraud")
plt.title("SVD Reconstruction Error 分布比較")
plt.xlabel("Reconstruction Error")
plt.legend()
plt.show()

# -------------------------------------------------------------
# 8. 顯示錯誤分數統計
# -------------------------------------------------------------
print("\n=== Reconstruction Error 統計 ===")
print(df.groupby("Class")["recon_error"].describe())

print("\n程式完成！")
```
