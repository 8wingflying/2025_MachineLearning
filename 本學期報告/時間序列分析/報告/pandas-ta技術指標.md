# 有誤
```python
import pandas_ta as ta

# 列出所有技術指標名稱
print(ta.indicators())

# 或查看所有分類
print(ta.categories)
```
## pandas-ta（Pandas Technical Analysis）
pandas-ta（Pandas Technical Analysis）內建 超過 150 種技術指標，涵蓋趨勢、動能、成交量、振盪、統計等多種類型。以下為完整分類總表（繁體中文）。

# pandas-ta 技術指標完整手冊（指標列表 + 函數對應）不對 ==> 有154個

> 本手冊根據 `pandas-ta` 官方套件（約 130+ 指標）之 Category 設定彙整，依 **功能類型** 分類，並給出：
>
> - **指標英文 / 中文名稱（簡稱）**
> - **對應函數名稱**（可用 `ta.xxx()` 或 `df.ta.xxx()` 呼叫）
> - **中文用途說明**（一句話重點）
>
> 實際可用指標仍以你安裝的版本為準，可用 `df.ta.indicators()` 自行列出。

---

## 0. 在自己環境列出全部 pandas-ta 指標

```python
import pandas as pd
import pandas_ta as ta

df = pd.DataFrame()          # 建一個空的 DataFrame
print(df.ta.indicators())    # 印出目前版本支援的所有指標名稱
```

若想看某一個指標的完整參數說明，例如 ADX：

```python
import pandas_ta as ta
help(ta.adx)          # 或 help(df.ta.adx)
```

---

## 1. 蜡燭圖（candles）

> 這些屬於 **K 線形態與特殊 K 線轉換**，多半需要安裝 TA-Lib 纔能使用完整形態庫。

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Candle Pattern | K 線形態偵測 | `ta.cdl_pattern(open, high, low, close, name="doji", length=10, scalar=0.1)` | 對應 TA-Lib 的多種 K 線形態（十字線、錘頭線等），輸出形態強度分數。 |
| Candle Z-Score | K 線 Z 分數 | `ta.cdl_z(open, high, low, close, length=10)` | 對 K 線實體與影線做標準化，觀察當前 K 線是否極端。 |
| Heikin Ashi | 平滑平均 K 線（HA） | `ta.ha(open, high, low, close)` | 將原始 K 線轉換成 Heikin Ashi，平滑價格波動，用於趨勢判讀。 |

---

## 2. 週期型（cycles）

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Even Better SineWave | 改良型正弦波週期指標 | `ta.ebsw(close, length=34)` | 嘗試捕捉價格中隱含的循環週期，用於研判循環高低點與震盪區間。 |

---

## 3. 動量與震盪指標（momentum）

> 主要衡量 **價格變動速度、相對強弱及超買超賣**。

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Awesome Oscillator | 強力震盪指標 | `ta.ao(high, low, fast=5, slow=34)` | 使用中間價的快慢均線差來衡量多空動能。 |
| Absolute Price Oscillator | 絕對價格震盪 | `ta.apo(close, fast=12, slow=26)` | 價格兩條均線的**差值**，觀察價格加速與減速。 |
| Bias | 乖離率 | `ta.bias(close, length=26)` | 價格相對移動平均線的偏離百分比，用於判斷過熱或過冷。 |
| Balance of Power | 多空力道 | `ta.bop(open, high, low, close)` | 以當日價位區間衡量多空力量誰佔優勢。 |
| BRAR | BRAR 情緒指標 | `ta.brar(open, high, low, close, length=26)` | 綜合比較開盤與高低價，衡量多空情緒強弱。 |
| Commodity Channel Index | CCI 頻道指標 | `ta.cci(high, low, close, length=20)` | 利用典型價格與平均偏離度衡量價格偏離程度。 |
| Chande Forecast Oscillator | CFO 預測震盪 | `ta.cfo(close, length=9)` | 比較當前價格與線性回歸預測價的差異。 |
| Center of Gravity | 重心指標 (CG) | `ta.cg(close, length=10)` | 將價格視為加權重心，找尋價格轉折點。 |
| Chande Momentum Oscillator | CMO | `ta.cmo(close, length=14)` | 上漲與下跌幅度的差值比率，類似 RSI 但對稱於 0。 |
| Coppock Curve | Coppock 曲線 | `ta.coppock(close, length=11)` | 原用於長期週期的多頭啟動偵測。 |
| Correlation Trend Indicator | CTI | `ta.cti(close, length=20)` | 利用價格與線性時間序列的相關係數判斷趨勢方向。 |
| Efficiency Ratio | 效率比 | `ta.er(close, length=10)` | 直線距離 / 路徑距離，比值越高代表走勢越趨直線。 |
| Elder Ray Index | Elder Ray 指標 | `ta.eri(high, low, close, length=13)` | 將價格與均線比較，拆成牛力與熊力兩條訊號。 |
| Fisher Transform | Fisher 轉換 | `ta.fisher(high, low, length=9)` | 將價格區間正規化到 -1~1，提高轉折點敏感度。 |
| Inertia | 慣性指標 | `ta.inertia(close, length=20)` | 結合 RSI 與趨勢概念，評估趨勢延續性。 |
| KDJ | KDJ 隨機指標 | `ta.kdj(high, low, close, length=9)` | 在 KD 基礎上加入 J 線，加強超買超賣信號。 |
| Know Sure Thing | KST 動量 | `ta.kst(close)` | 將多個 ROC 期別加權相加，做中長期動量判斷。 |
| MACD | 指數平滑異同移動平均 | `ta.macd(close, fast=12, slow=26, signal=9)` | 經典快慢 EMA 差值與訊號線，用於趨勢與反轉判讀。 |
| Momentum | 動量 | `ta.mom(close, length=10)` | 當前價與過去價的差值，最基本的動量衡量。 |
| Pretty Good Oscillator | PGO | `ta.pgo(close, length=14)` | 以 ATR 正規化的價格乖離，判斷極端位階。 |
| Percentage Price Oscillator | PPO | `ta.ppo(close, fast=12, slow=26)` | 快慢均線**百分比差**，類似 MACD 但具比例概念。 |
| Psychological Line | 心理線 | `ta.psl(close, length=12)` | 過去 N 日上漲日比例，反映市場情緒。 |
| Percentage Volume Oscillator | PVO | `ta.pvo(volume, fast=12, slow=26)` | 成交量均線的百分比差，衡量量能動能。 |
| Quantitative Qualitative Estimation | QQE | `ta.qqe(close, length=14)` | 在 RSI 基礎上加入平滑與波動帶，用於趨勢追蹤。 |
| Rate of Change | ROC 變動率 | `ta.roc(close, length=12)` | 價格相對 N 期前的百分比變化。 |
| Relative Strength Index | RSI 相對強弱指標 | `ta.rsi(close, length=14)` | 最常用超買超賣指標，0~100 區間。 |
| Relative Strength Xtra | RSX | `ta.rsx(close, length=14)` | 改良型 RSI，平滑且較不鋸齒。 |
| Relative Vigor Index | RVGI | `ta.rvgi(open, high, low, close, length=10)` | 比較收盤與開盤在區間中的位置來衡量趨勢。 |
| Slope | 斜率 | `ta.slope(close, length=10)` | 線性回歸的斜率，直接反映趨勢斜率大小。 |
| SMI Ergodic Indicator | SMI | `ta.smi(close, length=13)` | 改良型隨機指標，以中位價偏離作為基礎。 |
| Squeeze | 壓縮指標 | `ta.squeeze(high, low, close, bb_length=20, kc_length=20)` | 比較布林帶與肯特納通道，偵測低波動壓縮。 |
| Squeeze PRO | 進階壓縮指標 | `ta.squeeze_pro(high, low, close)` | 提供更多等級與濾波設定的壓縮訊號版本。 |
| Schaff Trend Cycle | STC | `ta.stc(close, tclength=10, fast=23, slow=50)` | 將 MACD 經過循環化，兼具趨勢與震盪特性。 |
| Stochastic Oscillator | 隨機指標 %K, %D | `ta.stoch(high, low, close, k=14, d=3)` | 用最高價/最低價區間衡量收盤價位置。 |
| Stochastic RSI | 隨機 RSI | `ta.stochrsi(close, length=14)` | 對 RSI 再做隨機化處理，放大極端訊號。 |
| TD Sequential | TD 序列 | `ta.td_seq(high, low, close)` | Tom DeMark 的計數型反轉指標。 |
| TRIX | 三重指數動量 | `ta.trix(close, length=15)` | 三次指數平滑後的 ROC，適合長趨勢。 |
| True Strength Index | TSI | `ta.tsi(close, long=25, short=13)` | 以雙層 EMA 平滑的價格變動率，去除雜訊。 |
| Ultimate Oscillator | UO | `ta.uo(high, low, close)` | 結合短中長期買盤壓力的綜合動量指標。 |
| Williams %R | 威廉指標 | `ta.willr(high, low, close, length=14)` | 類似隨機指標，以區間位置衡量超買超賣。 |

---

## 4. 重疊 / 均線類（overlap）

> 主要是各種 **移動平均、價位轉換與趨勢通道**。

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Arnaud Legoux Moving Average | ALMA | `ta.alma(close, length=9)` | 使用高斯權重的平滑均線，兼顧平滑與反應速度。 |
| Double Exponential Moving Average | DEMA | `ta.dema(close, length=20)` | 透過雙重 EMA 降低相位延遲。 |
| Exponential Moving Average | EMA 指數均線 | `ta.ema(close, length=20)` | 常用的指數加權移動平均線。 |
| Fibonacci Weighted Moving Average | FWMA | `ta.fwma(close, length=20)` | 以 Fibonacci 權重計算的加權均線。 |
| Gann High-Low Activator | HiLo | `ta.hilo(high, low, length=13)` | 以高低價移動平均形成趨勢通道。 |
| High-Low Average | HL2 | `ta.hl2(high, low)` | (High + Low) / 2 的中間價。 |
| Typical Price | HLC3 | `ta.hlc3(high, low, close)` | (High + Low + Close) / 3，技術分析常用典型價。 |
| Hull Moving Average | HMA | `ta.hma(close, length=21)` | 透過加權與開根減少延遲的快速均線。 |
| Ichimoku Kinko Hyo | 一目均衡表 | `ta.ichimoku(high, low, close)` | 日系多線指標：轉換線、基準線、雲帶與遲行線。 |
| Jurik Moving Average | JMA | `ta.jma(close, length=20)` | 目標是高平滑、低延遲的進階均線。 |
| Kaufman Adaptive Moving Average | KAMA | `ta.kama(close, length=10)` | 根據波動度自動調整平滑強度的自適應均線。 |
| Linear Regression | 線性回歸價格 | `ta.linreg(close, length=14)` | 用線性回歸擬合價格，輸出回歸線價。 |
| McGinley Dynamic | McGinley 動態均線 | `ta.mcgd(close, length=10)` | 隨波動自動調整速度的平滑線，減少假突破。 |
| Midpoint | 中點均價 | `ta.midpoint(close, length=2)` | 期間最高與最低價的中點。 |
| Midprice | 中位價 | `ta.midprice(high, low, length=2)` | 期間高低價中間水平，用於通道計算。 |
| OHLC4 | OHLC 平均價 | `ta.ohlc4(open, high, low, close)` | (O+H+L+C)/4，綜合四價平均。 |
| Pascals Weighted Moving Average | PWMA | `ta.pwma(close, length=20)` | 使用 Pascal 三角權重的加權均線。 |
| Wilder’s Moving Average | RMA | `ta.rma(close, length=14)` | RSI 等指標常用的 Wilder 平滑均線。 |
| Sine Weighted Moving Average | SINWMA | `ta.sinwma(close, length=20)` | 以正弦權重計算的均線。 |
| Simple Moving Average | SMA | `ta.sma(close, length=20)` | 最基本的算術平均移動均線。 |
| Super Smoother Filter | SSF | `ta.ssf(close, length=10)` | Ehlers 提出的超平滑濾波均線。 |
| Supertrend | Supertrend 趨勢線 | `ta.supertrend(high, low, close, length=10, multiplier=3)` | 結合 ATR 的跟隨型多空趨勢線。 |
| Symmetric Weighted Moving Average | SWMA | `ta.swma(close, length=20)` | 對稱權重形式的平滑均線。 |
| T3 Moving Average | T3 | `ta.t3(close, length=10, a=0.7)` | 多層 EMA 組成的高平滑趨勢線。 |
| Triple Exponential Moving Average | TEMA | `ta.tema(close, length=20)` | 三重 EMA 消除延遲的均線版本。 |
| Triangular Moving Average | TRIMA | `ta.trima(close, length=20)` | 以三角形權重計算的平滑均線。 |
| VIDYA | 變動指數動態均線 | `ta.vidya(close, length=14)` | 使用標準差或 CMO 作為可變權重的自適應均線。 |
| Volume Weighted Average Price | VWAP | `ta.vwap(high, low, close, volume)` | 成交量加權平均價格，常用於盤中支撐壓力。 |
| Volume Weighted Moving Average | VWMA | `ta.vwma(close, volume, length=20)` | 以成交量為權重的移動平均線。 |
| Weighted Closing Price | WCP | `ta.wcp(high, low, close)` | 加權收盤價，常作為價位代表值。 |
| Weighted Moving Average | WMA | `ta.wma(close, length=20)` | 線性權重型均線，近期資料權重較高。 |
| Zero Lag Moving Average | ZLMA | `ta.zlma(close, length=20)` | 透過相位補償降低延遲的均線。 |

---

## 5. 報酬 / 表現指標（performance）

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Log Return | 對數報酬率 | `ta.log_return(close, length=1, cumulative=False)` | 使用 ln 價格差計算的報酬率，可累積為長期報酬。 |
| Percent Return | 百分比報酬率 | `ta.percent_return(close, length=1, cumulative=False)` | 一般的百分比報酬，可做累積或單期分析。 |

---

## 6. 統計類指標（statistics）

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Entropy | 香農熵 | `ta.entropy(close, length=10)` | 衡量價格分布不確定度，值越高越雜亂。 |
| Kurtosis | 峰度 | `ta.kurtosis(close, length=30)` | 判斷報酬分布尾部厚度。 |
| Mean Absolute Deviation | 平均絕對離差 | `ta.mad(close, length=30)` | 價格相對平均的絕對偏離程度。 |
| Median | 中位數 | `ta.median(close, length=30)` | 區間內的中位價。 |
| Quantile | 分位數 | `ta.quantile(close, length=30, q=0.5)` | 區間內指定分位點的價格。 |
| Skew | 偏度 | `ta.skew(close, length=30)` | 報酬分布是否偏向左尾或右尾。 |
| Standard Deviation | 標準差 | `ta.stdev(close, length=30)` | 區間內價格波動大小。 |
| TOS All Stdev | Thinkorswim 全體標準差 | `ta.tos_stdevall(close, length=30)` | 對整體序列做標準差統計的特殊版本。 |
| Variance | 變異數 | `ta.variance(close, length=30)` | 標準差的平方，衡量波動能量。 |
| Z-Score | Z 分數 | `ta.zscore(close, length=30)` | 價格相對平均與標準差的位置，用於極端值判斷。 |

---

## 7. 趨勢指標（trend）

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Average Directional Index | ADX 趨向指標 | `ta.adx(high, low, close, length=14)` | 評估趨勢強度（非方向），值高代表趨勢明顯。 |
| Arnaud Legoux Moving Average Trend | AMAT 趨勢系統 | `ta.amat(close)` | 基於 ALMA 的多空切換趨勢策略指標。 |
| Aroon | Aroon 趨勢指標 | `ta.aroon(high, low, length=25)` | 比較近期高低點位置來判斷新趨勢。 |
| Choppiness Index | CHOP 震盪度 | `ta.chop(high, low, close, length=14)` | 越高代表盤整越明顯，越低代表趨勢越強。 |
| Chande Kroll Stop | CKSP 停損線 | `ta.cksp(high, low, close)` | 使用 ATR 與價格極值計算動態停損。 |
| Decay | 衰減函數 | `ta.decay(close, length=10)` | 以指數衰減加權過去價格，可當平滑工具。 |
| Decreasing | 連跌偵測 | `ta.decreasing(close, length=3)` | 判斷最近是否連續下跌。 |
| Detrended Price Oscillator | DPO 去趨勢價格 | `ta.dpo(close, length=20)` | 移除長期趨勢，聚焦短期波動。 |
| Increasing | 連漲偵測 | `ta.increasing(close, length=3)` | 判斷最近是否連續上漲。 |
| Long Run | 長期趨勢濾波 | `ta.long_run(close, length=100)` | 針對長周期的趨勢平滑版本。 |
| Parabolic SAR | PSAR | `ta.psar(high, low, close, step=0.02, max_step=0.2)` | 以拋物線方式跟隨價格的停損 / 趨勢線。 |
| Qstick | QStick | `ta.qstick(open, close, length=10)` | 以收盤減開盤平均判斷多空力量。 |
| Short Run | 短期趨勢濾波 | `ta.short_run(close, length=20)` | 針對短周期的趨勢版本，反應較快。 |
| Trend Signals | tsignals 趨勢訊號 | `ta.tsignals(close, asbool=True)` | 根據多個指標綜合產生買賣布林訊號。 |
| TTM Trend | TTM 趨勢 | `ta.ttm_trend(close)` | 延伸自 TTM 系列的多空色塊趨勢指標。 |
| Vertical Horizontal Filter | VHF | `ta.vhf(close, length=28)` | 比較直線距離與總波動，判斷盤整或趨勢。 |
| Vortex | Vortex 指標 | `ta.vortex(high, low, close, length=14)` | 使用高低價間距衡量多空力量變化。 |
| XSignals | xsignals 訊號 | `ta.xsignals(close, asbool=True)` | 替 Trend 類策略提供進出場訊號包裝。 |

---

## 8. 波動度指標（volatility）

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Aberration | Aberration 通道 | `ta.aberration(high, low, close, length=20)` | 結合均線與標準差的趨勢通道。 |
| Acceleration Bands | 加速帶 | `ta.accbands(high, low, close, length=20)` | 擴張型通道，試圖捕捉加速突破。 |
| Average True Range | ATR 平均真實波幅 | `ta.atr(high, low, close, length=14)` | 最常用的波動度衡量指標之一。 |
| Bollinger Bands | 布林通道 | `ta.bbands(close, length=20, std=2)` | 均線 ± 標準差的價格通道，含上下軌。 |
| Donchian Channel | 唐奇安通道 | `ta.donchian(high, low, length=20)` | 區間最高 / 最低價通道。 |
| High-Low Volatility Channel | HWC | `ta.hwc(high, low, close, length=20)` | 利用高低價區間構建的波動通道。 |
| Keltner Channel | 肯特納通道 | `ta.kc(high, low, close, length=20, scalar=2)` | 使用 ATR 寬度的 EMA 通道。 |
| Mass Index | MASS 指數 | `ta.massi(high, low, length=25)` | 利用高低價區間變化偵測趨勢反轉潛力。 |
| Normalized ATR | NATR | `ta.natr(high, low, close, length=14)` | ATR 以百分比形式呈現，便於跨商品比較。 |
| Price Distance | 價格距離 | `ta.pdist(close, length=14)` | 測量價格與某一基準線（如均線）的距離。 |
| Relative Volatility Index | RVI | `ta.rvi(close, length=14)` | 將標準差概念用 RSI 式表達的波動指標。 |
| Thermometer | Thermo 指標 | `ta.thermo(close, length=14)` | 用「溫度」概念視覺化近期波動程度。 |
| True Range | 真實波幅 | `ta.true_range(high, low, close)` | 單期最高 / 最低 / 前一收盤組合出的區間。 |
| Ulcer Index | UI 潰瘍指數 | `ta.ui(close, length=14)` | 專注於下跌幅度與持續時間的風險指標。 |

---

## 9. 成交量指標（volume）

| 英文名稱 | 中文名稱 | 函數 / 用法範例 | 中文說明 |
|---------|----------|-----------------|----------|
| Accumulation / Distribution | AD 累積/派發 | `ta.ad(high, low, close, volume)` | 將量價結合，估計資金累積或出場。 |
| Accumulation / Distribution Oscillator | ADOSC | `ta.adosc(high, low, close, volume)` | AD 的震盪版，用於短期量價訊號。 |
| Adjusted OBV | AOBV | `ta.aobv(close, volume)` | 變形版 OBV，以更精細邏輯處理量價。 |
| Chaikin Money Flow | CMF | `ta.cmf(high, low, close, volume, length=20)` | 一段期間內的量價加權，判斷買盤/賣盤。 |
| Elder’s Force Index | EFI 力量指數 | `ta.efi(close, volume, length=13)` | 價格變動乘上成交量，衡量做多 / 做空力量。 |
| Ease of Movement | EOM | `ta.eom(high, low, volume, length=14)` | 價格上漲「是否省力」，量小價漲為易動。 |
| Klinger Volume Oscillator | KVO | `ta.kvo(high, low, close, volume)` | 追蹤長期趨勢的量能震盪指標。 |
| Money Flow Index | MFI | `ta.mfi(high, low, close, volume, length=14)` | 價格與成交量結合的 RSI 版本。 |
| Negative Volume Index | NVI | `ta.nvi(close, volume)` | 只考慮量縮日變化的資金線。 |
| On-Balance Volume | OBV | `ta.obv(close, volume)` | 以上漲/下跌方向累計成交量。 |
| Positive Volume Index | PVI | `ta.pvi(close, volume)` | 只考慮量增日變化的資金線。 |
| Price-Volume | PVOL | `ta.pvol(close, volume)` | 價格與成交量直接相乘，可加上多空符號。 |
| Price Volume Rank | PVR | `ta.pvr(close, volume)` | 根據漲跌與量增減組合給 1~4 分類。 |
| Price Volume Trend | PVT | `ta.pvt(close, volume, drift=1)` | ROC 與成交量累積，用於長期資金流判斷。 |
| Volume Profile | VP | `ta.vp(close, volume, width=10)` | 價格區間上的量能分布（Volume by Price）。 |

---

## 10. 其他常見實用工具（部分版本可能另列於 utils）

> 以下不是傳統「技術指標」，但在策略中很常搭配使用（名稱可能隨版本調整）。

| 名稱 | 函數 / 用法範例 | 中文說明 |
|------|-----------------|----------|
| Binary To Number | `ta.bton(series)` | 將布林條件序列轉成整數編碼。 |
| Condition Counter | `ta.counter(condition, length=10)` | 對滿足條件的連續期數進行計數。 |
| Between | `ta.xbt(series, low, high)` | 判斷數值是否落在區間內。 |
| Condition Any | `ta.xex(series, length=5)` | 期間內是否「曾經」發生條件。 |
| Condition All | `ta.xev(series, length=5)` | 期間內是否「全部」都滿足條件。 |
| Dynamic Shift | `ta.xrf(series, length=5)` | 產生動態位移後的序列，方便對齊比較。 |
| Dynamic Rolling Highest | `ta.xhh(series, length=20)` | 計算期間最高值，常用作突破判斷。 |
| Dynamic Rolling Lowest | `ta.xfl(series, length=20)` | 計算期間最低值，常用作支撐判斷。 |

---

## 11. 小結與實戰建議

1. **確認版本差異**：不同版本的 `pandas-ta` 可能增加或調整少數函數名稱，建議先用 `df.ta.indicators()` 對照本表。  
2. **先選核心指標再擴充**：實務上不必一次用滿 100+ 指標，常見組合如：`SMA/EMA + RSI + MACD + ATR + OBV/MFI` 即可覆蓋多數情境。  
3. **多利用 DataFrame 擴充**：  
   ```python
   df.ta.rsi(length=14, append=True)
   df.ta.macd(append=True)
   df.ta.bbands(append=True)
   ```  
   讓指標直接變成多欄位 DataFrame，方便後續建模或回測。  

---

> ✅ 你可以直接下載本檔案，當作「pandas-ta 指標總表」隨時查詢，也可以自行補充參數說明與實戰範例。
