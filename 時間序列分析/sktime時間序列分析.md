# sktime 教學文件  
時間序列分析與預測入門指南

---

## 1. 什麼是 sktime？
- `sktime` 是一個專門處理「時間序列資料」的 Python 開源套件。
- 它把時間序列常見任務都做成統一 API，包含：
  - 預測（forecasting）  
  - 分類（time series classification）  
  - 分群 / 分段（clustering / segmentation）  
  - 特徵抽取（feature extraction）  
  - 時間序列轉換（transformers, sliding windows 等）
- 你可以把它想像成「時間序列版的 scikit-learn」：同樣有 `fit()` / `predict()` 的風格，但針對時序資料特化。

---

## 2. 為什麼用 sktime？(Chatgpt說的)
- 傳統的 pandas + sklearn 在做時間序列有幾個痛點：
  - 1. 資料結構不友善
  - 2. 預測 API 不統一  
  - 3. 時間步長 / 預測視窗（forecast horizon）處理麻煩  
- `sktime` 幫你解決這些問題：
  - 統一的 `fit()` / `predict(fh=...)` 介面  
  - 內建大量經典模型（ARIMA, Exponential Smoothing, Naive, Theta...）  
  - 支援回測（backtesting）與評估流程  

---

## 3. 安裝 sktime

```bash
pip install sktime
# 或完整安裝（含所有外部依賴）
pip install sktime[all-extras]
```

# 4. 時間序列資料格式

## 單一時間序列 (univariate)
```python

import pandas as pd

y = pd.Series(
    [100, 120, 130, 90, 110, 150],
    index=pd.period_range("2024-01-01", periods=6, freq="D")
)
```
## 多變量序列 (multivariate)
```python

Y = pd.DataFrame(
    {"temp": [21.0, 21.5, 22.1, 23.0],
     "humidity": [0.45, 0.47, 0.5, 0.52]},
    index=pd.period_range("2024-01-01", periods=4, freq="H")
)
```
多個時間序列 (panel / hierarchical)

使用 MultiIndex 來表示 (ID, 時間)。

## 5. Forecaster（預測器）範例
```python
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd

y = pd.Series(
    [100, 120, 130, 90, 110, 150],
    index=pd.period_range("2024-01-01", periods=6, freq="D")
)

fh = ForecastingHorizon([1, 2], is_relative=True)

forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y)
y_pred = forecaster.predict(fh)
print(y_pred)
```
# 6. 經典統計模型
## Exponential Smoothing
```python
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd

y = pd.Series(
    [120, 135, 150, 160, 145, 155, 170, 200],
    index=pd.period_range("2023-01", periods=8, freq="M")
)

fh = ForecastingHorizon([1, 2, 3], is_relative=True)

forecaster = ExponentialSmoothing(trend="add", seasonal="add", sp=12)
forecaster.fit(y)
print(forecaster.predict(fh))
```
## ARIMA
```python
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd

y = pd.Series(
    [112,118,132,129,121,135,148,148,136,119,104,118,
     115,126,141,135,125,149,170,170,158,133,114,140],
    index=pd.period_range("2022-01", periods=24, freq="M")
)

fh = ForecastingHorizon([1, 2, 3, 4, 5, 6], is_relative=True)
forecaster = ARIMA(order=(1,1,1))
forecaster.fit(y)
print(forecaster.predict(fh))
```
## 7. 訓練 / 測試切分與回測
```python
from sktime.forecasting.model_selection import temporal_train_test_split

y_train, y_test = temporal_train_test_split(y, test_size=12)
print(len(y_train), len(y_test))
```
## Walk-forward validation
```python
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.naive import NaiveForecaster

cv = SlidingWindowSplitter(fh=[1], window_length=12, step_length=1)
forecaster = NaiveForecaster(strategy="last")

scores = []
for train_idx, test_idx in cv.split(y):
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    forecaster.fit(y_train)
    pred = forecaster.predict(cv.get_test_fh())
    scores.append(mean_absolute_percentage_error(y_test, pred))

print(sum(scores)/len(scores))
```
## 8. Pipeline
```python
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Differencer
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

model = TransformedTargetForecaster(steps=[
    ("diff", Differencer(lags=1)),
    ("es", ExponentialSmoothing())
])
model.fit(y)
print(model.predict([1, 2, 3]))
```
## 9. 多步預測
```python
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd

y = pd.Series(
    [10,12,13,15,16,18,21,22,23,24,26,30],
    index=pd.period_range("2024-01", periods=12, freq="M")
)

fh = ForecastingHorizon([1,2,3,4,5,6], is_relative=True)
f = NaiveForecaster(strategy="drift")
f.fit(y)
print(f.predict(fh))
```
## 10. 評估指標
```python
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import pandas as pd

y_true = pd.Series([100,110,120], index=[1,2,3])
y_pred = pd.Series([105,108,123], index=[1,2,3])

print(mean_absolute_error(y_true, y_pred))
print(mean_absolute_percentage_error(y_true, y_pred))
```
## 11. 時間序列分類
```python
from sktime.classification.interval_based import TimeSeriesForestClassifier

clf = TimeSeriesForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
```
## 12. sktime vs 其他工具比較

工作內容	用 sktime 的理由	傳統作法

傳統 ARIMA	API 統一	statsmodels

模型比較	可共用回測流程	手刻

多步預測	fh 統一處理	for 迴圈

時間序列分類	原生支援	sklearn 不支援

GridSearch	ForecastingGridSearchCV	手刻調參
API 一致性	與 sklearn 類似	Prophet/ARIMA 不同

## 13. 注意事項
- Index 請使用 DatetimeIndex 或 PeriodIndex
- fh 可為相對步長或絕對時間
- 時序不可 shuffle
- 多變量與多系列格式不同，要區分清楚

## 14. 完整範例：從資料到評估

```python
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error

y = pd.Series(
    [100,110,105,120,130,128,140,150,160,158,170,175],
    index=pd.period_range("2024-01-01", periods=12, freq="D")
)

y_train, y_test = temporal_train_test_split(y, test_size=3)
fh = ForecastingHorizon(y_test.index, is_relative=False)

forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

print("真實值:\n", y_test)
print("預測值:\n", y_pred)
print("MAE:", mean_absolute_error(y_test, y_pred))
```

15. 延伸方向

ForecastingGridSearchCV 自動調參

多變量/階層式預測

加入外生變數 X 的預測

16. 總結
統一的時間序列框架

支援預測 / 分類 / 回測 / Pipeline

適合科研、教學與企業快速實驗
