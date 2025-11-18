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
from sklearn.ensemble import IsolationForest

# -----------------------------
# 1. Time Feature Engineering
# -----------------------------
df['hour'] = (df['Time'] / 3600) % 24
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Night anomaly feature
df['night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)

# -----------------------------
# 2. Amount Feature
# -----------------------------
df['LogAmount'] = np.log1p(df['Amount'])

# Standardize non-PCA features
scaler = StandardScaler()
df[['Amount_scaled', 'LogAmount_scaled']] = scaler.fit_transform(df[['Amount', 'LogAmount']])

# Amount outlier feature
df['amount_z'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['amount_outlier'] = (df['amount_z'] > 3).astype(int)

# -----------------------------
# 3. Select Features for Isolation Forest
# -----------------------------
features = ['hour_sin', 'hour_cos', 'LogAmount_scaled',
            'amount_outlier', 'night'] + [f'V{i}' for i in range(1, 29)]

X_if = df[features]

# -----------------------------
# 4. Train Isolation Forest
# -----------------------------
iso = IsolationForest(
    n_estimators=200,
    contamination=0.0017,  # 根據詐欺比例調整
    max_samples=256,
    random_state=42
)

iso.fit(X_if)

df['anomaly_score'] = iso.decision_function(X_if)
df['anomaly_label'] = iso.predict(X_if)    # -1 = 異常, 1 = 正常

# Convert to 0/1 label
df['anomaly'] = df['anomaly_label'].map({1: 0, -1: 1})

df[['anomaly_score', 'anomaly', 'Class']].head()
```
