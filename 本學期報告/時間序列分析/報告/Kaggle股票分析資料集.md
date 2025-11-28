## Kaggle 股票分析資料集精選 (Stock Analysis Datasets)

這份清單整理了 Kaggle 上針對股市分析、演算法交易與金融情緒分析的高質量資料集。

---

## 1. 歷史股價數據 (Time Series & Historical Data)
這類資料集包含標準的 OHLCV (Open, High, Low, Close, Volume)，適合進行趨勢預測、技術指標計算與回測。

### **Huge Stock Market Dataset**
* **簡介**: Kaggle 上數據量最龐大的資料集之一，包含數千檔美股與 ETF 的每日歷史數據。
* **格式**: 每個股票為獨立的 `.txt` / `.csv` 檔案。
* **適用**: 大規模市場掃描、板塊分析、長期趨勢預測。
* **關鍵字搜尋**: `Huge Stock Market Dataset` (by Boris Marjanovic)
* [Huge Stock Market Dataset | Kaggle](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)

### **S&P 500 Stock Data**
* **簡介**: 專注於標普 500 指數成分股的歷史數據，數據相對乾淨標準。
* **適用**: 投資組合優化 (Portfolio Optimization)、相關性矩陣分析。
* **關鍵字搜尋**: `S&P 500 stock data` (by Cam Nugent)
* [S&P 500 stock data | Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500)
* https://github.com/MelikBLK/SP500-Data-Analytics-and-Price-Prediction/tree/main

### **Nasdaq & NYSE Stocks (Fundamental)**
* **簡介**: 除了價格外，部分資料集包含基本面指標 (PE Ratio, EPS, Market Cap)。
* **適用**: 價值投資策略、多因子模型 (Multi-factor Model)。
* [NASDAQ and NYSE stocks histories | Kaggle](https://www.kaggle.com/datasets/qks1lver/nasdaq-and-nyse-stocks-histories)

---

## 2. 新聞與情緒分析 (NLP & Sentiment Analysis)
這類資料集用於訓練模型解讀市場情緒，是 AI FinTech 的熱門領域。
- [利用 LangChain 分析金融新聞的情緒](https://patotricks15.medium.com/sentiment-analysis-of-financial-news-using-langchain-43b39eb401a7)
- 202410 [Financial Sentiment Analysis on News and Reports Using Large Language Models and FinBERT](https://arxiv.org/abs/2410.01987)
- 2024 [A FinBERT Framework for Sentiment Analysis of Chinese Financial News](https://ieeexplore.ieee.org/document/10699096)

### **Daily News for Stock Market Prediction**
* **簡介**: 經典入門資料集。結合了 DJIA (道瓊指數) 的漲跌標籤與當日 Reddit WorldNews 的熱門頭條。
* **適用**: 二元分類 (預測漲跌)、文本情感分析。
* **關鍵字搜尋**: `Daily News for Stock Market Prediction` (by Aaron7sun)
* [Daily News for Stock Market Prediction | Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews)

### **Financial Sentiment Analysis**
* **簡介**: 包含標註好情緒 (Positive/Negative/Neutral) 的財經新聞語句。
* **適用**: 訓練 FinBERT 模型、特定財經術語的情緒判讀。
* **關鍵字搜尋**: `Sentiment Analysis for Financial News`
* [Sentiment Analysis for Financial News | Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

### **Elon Musk Tweets & Stock Impact**
* **簡介**: 收錄 Elon Musk 推文與 Tesla/Dogecoin 的同期價格。
* **適用**: 影響力分析、事件驅動策略 (Event-driven strategy)。

---

## 3. 進階與特定市場 (Advanced & Local)

### **Taiwan Stock Exchange Data (TWSE)**
* **簡介**: 包含台股加權指數或熱門個股 (如 TSMC) 的歷史數據。
* **注意**: 建議檢查資料集的 "Last Updated" 日期，台股建議搭配 `yfinance` 或 `twstock` 抓取最新資料。
* **關鍵字搜尋**: `Taiwan Stock Price` 或 `TWSE`

### **G-Research Crypto Forecasting**
* **簡介**: 來自大型競賽的加密貨幣分鐘級數據。
* **適用**: 高頻交易 (HFT) 模擬、微觀結構分析。

---

## 4. 推薦 Python 工具庫

在處理上述資料時，建議搭配以下工具：

| 工具庫 | 用途 |
| :--- | :--- |
| **Pandas** | 數據清洗、時間序列處理 (DataFrame) |
| **yfinance** | 獲取最新即時股價 (彌補 Kaggle 資料滯後問題) |
| **TA-Lib** | 計算技術指標 (RSI, MACD, Bollinger Bands) |
| **Backtrader** | 策略回測框架 |
| **Scikit-learn / PyTorch** | 機器學習與深度學習模型構建 |

---

> **Note**: Kaggle 資料集多為靜態歷史數據。若需開發實盤交易機器人，請務必串接即時 API (如 Yahoo Finance API, Alpha Vantage, 或券商 API)。
