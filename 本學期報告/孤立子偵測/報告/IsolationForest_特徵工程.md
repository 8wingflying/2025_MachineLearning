# Isolation Forest 特徵工程
- 為異常偵測最佳化，不使用 SMOTE、Class Weight、不做 label-based selection
- 比較有進行特徵工程與沒用特徵工程有何用?
- 建議特徵：

📌 使用

V1 ~ V28（原 PCA 特徵）

hour_sin, hour_cos（週期）

LogAmount_scaled（主要金額特徵）

amount_outlier（少量離群點特徵）

night（時間異常特徵）

❌ 不使用

SMOTE / 重採樣（無監督）

One-hot（易造成維度偏差）

Time（秒數沒意義）

```python

```
