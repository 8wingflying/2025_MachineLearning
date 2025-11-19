# 有誤
```python
import pandas_ta as ta

# 列出所有技術指標名稱
print(ta.indicators())

# 或查看所有分類
print(ta.categories)
```
## 
pandas-ta（Pandas Technical Analysis）內建 超過 150 種技術指標，涵蓋趨勢、動能、成交量、振盪、統計等多種類型。以下為完整分類總表（繁體中文）。

🧭 1. 趨勢類（Trend Indicators）
指標	說明
SMA	Simple Moving Average（簡單移動平均）
EMA	Exponential Moving Average（指數移動平均）
WMA	Weighted Moving Average（加權移動平均）
HMA	Hull Moving Average
DEMA	Double Exponential MA
TEMA	Triple Exponential MA
KAMA	Kaufman Adaptive MA
ZLEMA	Zero-Lag EMA
T3	T3 Moving Average
ALMA	Arnaud Legoux MA
VWMA	Volume-Weighted MA
LSMA / TSF	Least Square MA（時間序列預測線）
MCGD	McGinley Dynamic
SAR	Parabolic SAR
PSAR	Parabolic Stop and Reverse
Ichimoku	一目均衡表（雲圖）
SuperTrend	超級趨勢指標
Trendlines	自動趨勢線（部分版本支持）
🔥 2. 動能類（Momentum Indicators）
指標	說明
RSI	相對強弱指標
Stoch	KD 隨機指標
StochRSI	隨機 RSI
MFI	Money Flow Index（資金流量）
CCI	Commodity Channel Index
ROC	Rate of Change（變動率）
MOM	Momentum
PPO	Percentage Price Oscillator
APO	Absolute Price Oscillator
KDJ	隨機 KDJ
TRIX	Triple Exponential Rate of Change
UO	Ultimate Oscillator
AO	Awesome Oscillator
BOP	Balance of Power
PSAR	Parabolic SAR（也可算趨勢）
Squeeze / Squeeze Pro	趨勢轉折偵測
ERI	Elder Ray Index
📉 3. 振盪類（Oscillators）
指標
MACD
MACD Histogram
MACD Signal
%B（Bollinger Percent）
BBW（Bollinger Band Width）
WaveTrend（WT）
Fisher Transform
Schaff Trend Cycle（STC）
📊 4. 波動度類（Volatility）
指標
ATR（Average True Range）
True Range（TR）
Bollinger Bands
Donchian Channels
Keltner Channels
STDEV（標準差）
Ulcer Index
Mass Index
Normalized ATR
📈 5. 成交量類（Volume Indicators）
指標
OBV（On-Balance Volume）
VWAP（Volume Weighted Average Price）
AD（Accumulation/Distribution）
ADL（A/D Line）
CMF（Chaikin Money Flow）
NVI（Negative Volume Index）
PVI（Positive Volume Index）
VZO（Volume Zone Oscillator）
EMV（Ease of Movement）
MFI（資金流量指標）
🧮 6. 統計類（Statistical Indicators）
指標
Z-Score
Entropy（香農熵）
Kurtosis（峰度）
Skew（偏度）
Linear Regression（線性迴歸）
Rolling Regression
Correlation（自相關/互相關）
Covariance
Median Filter
Quantile Bands
Percentile Channel
##📦 7. 其他複合型（Misc）指標
- Heikin-Ashi（平均 K 線）
Renko
Pivot Points（支撐/壓力）
Fractals（分形）
ZigZag（之字轉折）
Log Returns
Increasing/Decreasing Count
Multi-Indicator Strategies（如 Alligator）
