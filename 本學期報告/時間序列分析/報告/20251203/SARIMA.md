## SARIMA(Gemini)
- https://www.geeksforgeeks.org/deep-learning/time-series-modeling-with-statsmodels/
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# 設定繪圖風格
plt.style.use('ggplot')

# ==========================================
# 1. 生成假數據 (Synthetic Data Generation)
# ==========================================
def generate_data(n_samples=200):
    np.random.seed(42)
    # 時間索引
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='M')
    
    # 線性趨勢 (Trend)
    trend = np.linspace(0, 10, n_samples)
    
    # 季節性 (Seasonality, 週期為 12 個月)
    seasonality = 10 * np.sin(np.linspace(0, 3.14 * 8, n_samples))
    
    # 雜訊 (Noise)
    noise = np.random.normal(0, 2, n_samples)
    
    # 合成數據
    data = trend + seasonality + noise + 50
    return pd.Series(data, index=dates)

# 生成數據
ts_data = generate_data()

# 繪製原始數據
plt.figure(figsize=(12, 6))
plt.plot(ts_data)
plt.title('Simulated Time Series Data (Trend + Seasonality)')
plt.show()

# ==========================================
# 2. 平穩性檢定 (Stationarity Check - ADF Test)
# ==========================================
def adf_check(time_series):
    result = adfuller(time_series)
    print('--- Augmented Dickey-Fuller Test ---')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("結論: 數據是平穩的 (Reject Null Hypothesis)")
    else:
        print("結論: 數據是非平穩的 (Fail to Reject Null Hypothesis)")
    print('------------------------------------')

adf_check(ts_data)
# 注意：由於我們的數據有趨勢，通常 ADF 會顯示不平穩，這暗示我們需要進行差分 (Integration, d=1)

# ==========================================
# 3. SARIMA 建模 (Modeling)
# ==========================================
# 拆分訓練集與測試集 (最後 12 個月做測試)
train_data = ts_data[:-12]
test_data = ts_data[-12:]

# 設定 SARIMA 參數
# order=(p, d, q): 非季節性參數
# seasonal_order=(P, D, Q, s): 季節性參數 (s=12 代表年週期)
# 這裡我們手動設定參數，實際專案中可用 auto_arima 尋找最佳參數
model = SARIMAX(train_data, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# 顯示模型統計摘要
print(results.summary())

# ==========================================
# 4. 模型診斷 (Diagnostics)
# ==========================================
# 檢查殘差是否為常態分佈且無相關性
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# ==========================================
# 5. 預測與評估 (Forecasting & Evaluation)
# ==========================================
# 預測測試集區間
forecast_object = results.get_forecast(steps=len(test_data))
forecast = forecast_object.predicted_mean
conf_int = forecast_object.conf_int()

# 計算 MSE
mse = mean_squared_error(test_data, forecast)
print(f'\nMean Squared Error on Test Data: {mse:.4f}')

# 繪製最終結果
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Train')
plt.plot(test_data.index, test_data, label='Test (Actual)', color='green')
plt.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='--')

# 繪製信賴區間 (Confidence Interval)
plt.fill_between(forecast.index, 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='pink', alpha=0.3, label='95% Confidence Interval')

plt.title('SARIMA Model Forecast vs Actuals')
plt.legend()
plt.show()
```

```
這段程式碼主要是在進行時間序列分析與預測，
使用的是 SARIMA (Seasonal AutoRegressive Integrated Moving Average) 模型。

它分析的資料集是澳洲抗糖尿病藥物的處方數量 (AusAntidiabeticDrug.csv)。

以下是程式碼各個部分的詳細解說：

1. 匯入套件與資料讀取
匯入庫：使用了 pandas, numpy 處理資料，matplotlib 繪圖，以及 statsmodels 進行統計分析（如 
SARIMAX, STL 分解, adfuller 檢定等）。
讀取資料：讀取 ../data/AusAntidiabeticDrug.csv，並查看資料的前後幾筆與形狀。


2. 資料視覺化 (Visualization)
繪製原始的時間序列圖，X 軸為日期，Y 軸為藥物處方數量。
設定了 X 軸的刻度與標籤格式，並將圖表存為圖片。


3. 資料探索 (Exploration)
STL 分解：使用 STL 將時間序列分解為三個部分：
Trend (趨勢)：長期的增長或減少趨勢。
Seasonal (季節性)：固定週期的波動（這裡設定週期 period=12，即一年）。
Residuals (殘差)：扣除趨勢和季節性後剩下的雜訊。
將這四個部分（原始、趨勢、季節、殘差）繪製在同一張圖中觀察。


4. 建模前置作業 (Modeling)
定態性檢定 (ADF Test)：使用 Augmented Dickey-Fuller 檢定來檢查資料是否為定態 (Stationary)。
先檢定原始資料（通常不平穩）。
進行一階差分 (First Difference, $d=1$) 後再檢定。
再進行季節性差分 (Seasonal Difference, $D=1$) 後再檢定。

結論：確定了差分參數 $d=1$ 和 $D=1$，且因為是月資料，季節週期 $m=12$。
訓練/測試集分割：前 168 筆資料作為訓練集 (Train)，剩下的作為測試集 (Test)。


5. SARIMA 模型訓練
參數最佳化：定義了 optimize_SARIMAX函數，透過網格搜索 (Grid Search) 嘗試不同的 $(p, q, P, Q)$ 組合，並根據 AIC (Akaike Information Criterion) 指標來選出最好的模型（AIC 越低越好）。

模型擬合：選定參數 order=(2,1,3) 和 seasonal_order=(1,1,3,12) 建立 SARIMA 模型並進行訓練。

殘差診斷：
plot_diagnostics：檢查殘差是否符合常態分佈且無相關性。
acorr_ljungbox：使用 Ljung-Box 檢定確認殘差是否為白雜訊 (White Noise)。


6. 預測 (Forecasting)
滾動預測 (Rolling Forecast)：定義 rolling_forecast函數，模擬實際場景。
每預測一個窗口 (Window) 的未來值後，就將實際觀測值加入訓練資料中，重新訓練或更新模型，再預測下一個窗口。

比較兩種方法：
last_season (Naive Seasonal)：單純假設未來的數值等於上一季（去年同期）的數值，作為基準線 (Baseline)。
SARIMA：使用訓練好的 SARIMA 模型進行預測。
繪圖比較：將實際值 (Actual)、Naive Seasonal 預測、SARIMA 預測畫在同一張圖上比較。


7. 評估 (Evaluate)
MAPE (Mean Absolute Percentage Error)：計算平均絕對百分比誤差來評估模型準確度。
結果比較：計算並印出兩種方法的 MAPE，並繪製長條圖比較。
通常 SARIMA 的誤差會比 Naive 方法低，代表模型有效。
```
