#### [Python 時間序列預測](https://www.tenlong.com.tw/products/9787111754466?list_name=srh)
- [Time Series Forecasting in Python](https://learning.oreilly.com/library/view/time-series-forecasting/9781617299889/)
- https://github.com/marcopeix/TimeSeriesForecastingInPython
## Part 1. Time waits for no one
```
1 Understanding time series forecasting
2 A naive prediction of the future
3 Going on a random walk
```
## Part 2. Forecasting with statistical models
```
4 Modeling a moving average process
5 Modeling an autoregressive process
6 Modeling complex time series
7 Forecasting non-stationary time series
8 Accounting for seasonality
9 Adding external variables to our model
10 Forecasting multiple time series
11 Capstone: Forecasting the number of antidiabetic drug prescriptions in Australia
```
## Part 3. Large-scale forecasting with deep learning
```
12 Introducing deep learning for time series forecasting
13 Data windowing and creating baselines for deep learning
14 Baby steps with deep learning
15 Remembering the past with LSTM
16 Filtering a time series with CNN
17 Using predictions to make more predictions
18 Capstone: Forecasting the electric power consumption of a household
```
## Part 4. Automating forecasting at scale
```
19 Automating time series forecasting with Prophet
20 Capstone: Forecasting the monthly average retail price of steak in Canada
21 Going above and beyond
Appendix. Installation instructions
```

```
第1章　瞭解時間序列預測　 3
1.1　時間序列簡介　 4
1.2　時間序列預測概覽　 7
1.2.1　設定目標　 8
1.2.2　確定預測對象　 8
1.2.3　設定預測範圍　 8
1.2.4　收集資料　 8
1.2.5　開發預測模型　 8
1.2.6　部署到生產中　 9
1.2.7　監控　 9
1.2.8　收集新的資料　 9
1.3　時間序列預測與其他迴歸任務的差異　 10
1.3.1　時間序列有順序　 10
1.3.2　時間序列有時沒有特徵　 10
1.4　下一步　 11
第2章　對未來的簡單預測　 12
2.1　定義基準模型　 13
2.2　預測歷史均值　 14
2.2.1　基線實現準備　 15
2.2.2　實作歷史均值基準　 16
2.3　預測最後一年的平均值　 19
2.4　使用最後已知數值進行預測　 21
2.5　實現簡單的季節性預測　 22
2.6　下一步　 23
第3章　來一次隨機遊走　 25
3.1　隨機遊走過程　 26
3.2　辨識隨機遊走　 29
3.2.1　平穩性　 29
3.2.2　平穩性檢定　 31
3.2.3　自相關函數　 34
3.2.4　把它們組合在一起　 34
3.2.5　GOOGL是隨機遊走嗎　 37
3.3　預測隨機遊走　 39
3.3.1　長期預測　 39
3.3.2　預測下一個時間步長　 44
3.4　下一步　 46
3.5　練習　 46
3.5.1　模擬與預測隨機遊走　 46
3.5.2　預測GOOGL的每日收盤價　 47
3.5.3　預測你所選擇的股票的每日收盤價　 47
第二部分　使用統計模型進行預測
第4章　移動平均過程建模　 51
4.1　定義移動平均過程　 52
4.2　預測移動平均過程　 57
4.3　下一步　 64
4.4　練習　 65
4.4.1　模擬MA(2)過程並做預測　 65
4.4.2　模擬MA(q)過程並做預測　 65
第5章　自迴歸過程建模　 67
5.1　預測零售店平均每週客流量　 67
5.2　定義自迴歸過程　 69
5.3　求平穩自迴歸過程的階數　 70
5.4　預測自迴歸過程　 76
5.5　下一步　 82
5.6　練習　 82
5.6.1　模擬AR(2)過程並做預測　 82
5.6.2　模擬AR(p)過程並做預測　 83
第6章　複雜時間序列建模　 84
6.1　預測資料中心頻寬使用量　 85
6.2　研究自回歸移動平均過程　 86
6.3　確定一個平穩的ARMA過程　 88
6.4　設計一個通用的建模過程　 91
6.4.1　瞭解AIC　 92
6.4.2　使用AIC選擇模型　 93
6.4.3　瞭解殘差分析　 95
6.4.4　進行殘差分析　 99
6.5　應用通用建模過程　 102
6.6　預測頻寬使用情況　 108
6.7　下一步　 112
6.8　練習　 113
6.8.1　對模擬的ARMA(1,1)過程進行預測　 113
6.8.2　模擬ARMA(2,2)過程並進行預測　 113
第7章　非平穩時間序列預測　 115
7.1　定義差分自迴歸移動平均模型　 116
7.2　修改通用建模流程以考慮非平穩序列　 117
7.3　預測一個非平穩時間序列　 119
7.4　下一步　 125
7.5　練習　 126
第8章　考慮季節性　 127
8.1　研究SARIMA(p,d,q)(P,D,Q)m模型　 128
8.2　辨識時間序列的季節性模式　 129
8.3　預測航空公司每月乘客數　 133
8.3.1　使用ARIMA(p,d,q)模型進行預測　 135
8.3.2　使用SARIMA(p,d,q)(P,D,Q)m模型進行預測　 139
8.3.3　比較每種預測方法的效能　 142
8.4　下一步　 144
8.5　練習　 145
第9章　為模型新增外生變量　 146
9.1　研究SARIMAX模型　 147
9.1.1　探討美國宏觀經濟資料集的外生變量　 148
9.1.2　使用SARIMAX的註意事項　 150
9.2　使用SARIMAX模型預測實際GDP　 151
9.3　下一步　 158
9.4　練習　 159
第10章　預測多變量時間序列　 160
10.1　研究VAR模型　 161
10.2　設計VAR(p)建模過程　 163
10.3　預測實際可支配所得與實際消費　 164
10.4　下一步　 174
10.5　練習　 174
10.5.1　使用VARMA模型預測realdpi和realcons　 174
10.5.2　使用VARMAX模型預測realdpi和realcons　 175
第11章　頂點計畫：預測澳大利亞
抗糖尿病藥物處方的數量　 176
11.1　導入所需的庫並加載資料　 177
11.2　可視化序列及其分量　 178
11.3　對資料進行建模　 180
11.3.1　進行模型選擇　 181
11.3.2　進行殘差分析　 183
11.4　預測與評估模型的表現　 184
11.5　下一步　 187
第三部分　使用深度學習進行大規模預測
第12章　將深度學習引入時間序列預測　 191
12.1　何時使用深度學習進行時間序列預測　 191
12.2　探索不同類型的深度學習模型　 192
12.3　準備應用深度學習進行預測　 194
12.3.1　進行資料探
```
