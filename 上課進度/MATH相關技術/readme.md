## optimization
- dual ==> SVM
## Linear Algebra
- SVD
- PCA
- NMF

## 範例
```python
import numpy as np
import scipy.linalg as la

# 0. 定義係數矩陣 A 和常數向量 B
# 方程式：
#  2x +  y -  z =   8
# -3x -  y + 2z = -11
# -2x +  y + 2z =  -3

A = np.array([
    [ 2,  1, -1],
    [-3, -1,  2],
    [-2,  1,  2]
])

B = np.array([8, -11, -3])

print(f"係數矩陣 A:\n{A}")
print(f"常數向量 B:\n{B}\n")
print("-" * 30)

# ==========================================
# 方法一 & 二：加減消去法 / 代入消去法 (Standard Solver)
# 說明：在數值運算庫中，這兩者對應的是標準的線性方程求解函式。
# Numpy 使用 np.linalg.solve
# ==========================================
print("【方法一 & 二：標準求解器 (對應消去法/代入法)】")

try:
    # np.linalg.solve 底層通常使用 LAPACK 的 LU 分解
    x_solve = np.linalg.solve(A, B)
    print(f"解 (x, y, z): {x_solve}")
    # 驗證
    print(f"驗證 Ax - B (應接近 0): {np.allclose(np.dot(A, x_solve), B)}")
except np.linalg.LinAlgError:
    print("矩陣不可逆，無法求解")

print("-" * 30)

# ==========================================
# 方法三：高斯消去法 (Gaussian Elimination / LU Decomposition)
# 說明：Scipy 的 lu 函式可以將 A 分解為 P(列交換), L(下三角), U(上三角)
# 求解過程變成兩個步驟：Ly = P*B (前代), Ux = y (回代)
# ==========================================
print("【方法三：高斯消去法 (利用 Scipy LU 分解)】")

# 1. 進行 LU 分解: A = P * L * U
P, L, U = la.lu(A)

print(f"L (下三角矩陣):\n{L}")
print(f"U (上三角矩陣 - 即高斯消去後的梯形矩陣):\n{U}")

# 2. 求解
# 由於 A x = B => P L U x = B => L U x = P_inv B
# 令 U x = y，則 L y = P_inv B (或是 P.T @ B，因為 P 是正交矩陣)
PB = np.dot(P.T, B) # 對常數項進行同樣的列交換
y = la.solve_triangular(L, PB, lower=True)  # 解 Ly = PB
x_lu = la.solve_triangular(U, y, lower=False) # 解 Ux = y (回代)

print(f"解 (x, y, z): {x_lu}")

print("-" * 30)

# ==========================================
# 方法四：克拉瑪公式 (Cramer's Rule)
# 說明：利用行列式 (Determinant) 求解
# x_i = det(A_i) / det(A)
# ==========================================
print("【方法四：克拉瑪公式 (Determinant)】")

# 1. 計算主行列式 delta
det_A = np.linalg.det(A)
print(f"Delta (det(A)): {det_A:.2f}")

if not np.isclose(det_A, 0):
    # 2. 準備置換後的矩陣
    A_x = A.copy()
    A_x[:, 0] = B  # 第一行換成 B
    
    A_y = A.copy()
    A_y[:, 1] = B  # 第二行換成 B
    
    A_z = A.copy()
    A_z[:, 2] = B  # 第三行換成 B
    
    # 3. 計算各分量行列式
    det_x = np.linalg.det(A_x)
    det_y = np.linalg.det(A_y)
    det_z = np.linalg.det(A_z)
    
    # 4. 求解
    x_cramer = det_x / det_A
    y_cramer = det_y / det_A
    z_cramer = det_z / det_A
    
    print(f"Delta_x: {det_x:.2f}, Delta_y: {det_y:.2f}, Delta_z: {det_z:.2f}")
    print(f"解 (x, y, z): [{x_cramer:.0f}. {y_cramer:.0f}. {z_cramer:.0f}.]")
else:
    print("Delta 為 0，無法使用克拉瑪公式")

print("-" * 30)

# ==========================================
# 方法五：反矩陣法 (Inverse Matrix)
# 說明：X = A^(-1) * B
# ==========================================
print("【方法五：反矩陣法 (Inverse Matrix)】")

try:
    # 1. 計算反矩陣
    A_inv = np.linalg.inv(A)
    print("A 的反矩陣:\n", A_inv)
    
    # 2. 矩陣相乘
    x_inv = np.dot(A_inv, B)
    
    print(f"解 (x, y, z): {x_inv}")
except np.linalg.LinAlgError:
    print("矩陣不可逆，無反矩陣")
```
