#
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 先上傳
df = pd.read_csv("creditcard.csv")

# 基本資訊
print(df.head())
print(df.describe()) # 說明描述統計的基本資料
print(df['Class'].value_counts()) # 正常與異常筆數
print(df['Class'].value_counts(normalize=True))

# 類別分布圖
sns.countplot(data=df, x='Class')
plt.title("Class Distribution")
plt.show()

# Amount 分布
plt.figure(figsize=(6,4))
sns.histplot(df['Amount'], bins=50)
plt.title("Amount Distribution")
plt.show()

# Time 分布
plt.figure(figsize=(6,4))
sns.histplot(df['Time'], bins=50)
plt.title("Time Distribution")
plt.show()

# Correlation[Pearson Correlation Coefficient] == Heatmap
plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", vmax=0.3)
plt.title("Correlation Matrix")
plt.show()

# 比較詐欺與正常交易的 Amount
plt.figure(figsize=(6,4))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Amount by Class")
plt.show()

```
