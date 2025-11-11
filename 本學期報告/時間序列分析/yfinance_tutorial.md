# 📊 yfinance 教學文件  
> 使用 Python 快速抓取 Yahoo Finance 股票與財經資料  

---

## 🧩 1️⃣ 安裝與載入套件
```bash
pip install yfinance
```
```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
```

---

## 💡 2️⃣ 下載股價資料
```python
# 抓取台積電(TSM) 2024 年資料
tsm = yf.download("TSM", start="2024-01-01", end="2024-12-31")
print(tsm.head())
```

| 欄位 | 說明 |
|------|------|
| Open | 開盤價 |
| High | 最高價 |
| Low  | 最低價 |
| Close | 收盤價 |
| Adj Close | 調整後收盤價（含股息與拆股修正） |
| Volume | 交易量 |

---

## 📈 3️⃣ 視覺化股價走勢
```python
plt.figure(figsize=(10,5))
plt.plot(tsm["Close"], label="TSMC Close Price")
plt.title("TSMC Stock Price 2024")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
```

---

## 🧮 4️⃣ 技術分析：移動平均線 (Moving Average)
```python
tsm["MA20"] = tsm["Close"].rolling(window=20).mean()
tsm["MA50"] = tsm["Close"].rolling(window=50).mean()

plt.figure(figsize=(10,5))
plt.plot(tsm["Close"], label="Close Price", color="gray")
plt.plot(tsm["MA20"], label="20-Day MA", color="blue")
plt.plot(tsm["MA50"], label="50-Day MA", color="orange")
plt.title("TSMC Moving Averages")
plt.legend()
plt.show()
```

---

## 💬 5️⃣ 單一公司資訊
```python
ticker = yf.Ticker("TSM")
print(ticker.info["longName"])
print(ticker.info["sector"])
print(ticker.info["marketCap"])
```

---

## 🧾 6️⃣ 財報資料（損益表、資產負債表、現金流量表）
```python
income_stmt = ticker.financials
balance_sheet = ticker.balance_sheet
cashflow = ticker.cashflow

print("損益表：")
print(income_stmt.head())
```

---

## 🌏 7️⃣ 多檔股票同時下載
```python
data = yf.download(["AAPL", "MSFT", "GOOG"], start="2024-01-01", end="2024-12-31")["Adj Close"]
data.plot(figsize=(10,5), title="Tech Stocks 2024 Performance")
plt.show()
```

---

## 🪙 8️⃣ 計算報酬率
```python
tsm["Daily_Return"] = tsm["Close"].pct_change()
cumulative_return = (1 + tsm["Daily_Return"]).cumprod() - 1
cumulative_return.plot(title="TSMC Cumulative Return 2024")
plt.show()
```

---

## 📊 9️⃣ 股息與拆股資料
```python
print("股息紀錄：")
print(ticker.dividends.tail())

print("拆股紀錄：")
print(ticker.splits.tail())
```

---

## 🤠 🔟 範例：台灣加減指數與個股比較
```python
twii = yf.download("^TWII", start="2024-01-01", end="2024-12-31")["Adj Close"]
tsmc = yf.download("TSM", start="2024-01-01", end="2024-12-31")["Adj Close"]

compare = pd.DataFrame({"TSMC": tsmc, "TAIEX": twii})
compare_normalized = compare / compare.iloc[0]
compare_normalized.plot(figsize=(10,5), title="TSMC vs TAIEX Performance 2024")
plt.show()
```

---

## 🦯 延伸應用
| 主題 | 範例 |
|------|------|
| 技術分析 | 結合 TA-Lib 進行 RSI、MACD 等指標 |
| 財務比率分析 | EPS、ROE、ROA 等自動化計算 |
| 投資組合風險分析 | 與 NumPy、PyPortfolioOpt 整合 |
| 即時股價 | 使用 `yf.Ticker("TSM").fast_info` 或 `yf.download(interval="1m")` |

---

## 📚 參考資源
- [yfinance 官方文件](https://github.com/ranaroussi/yfinance)
- [Yahoo Finance API Reference](https://finance.yahoo.com/)
- [Pandas DataFrame 操作教學](https://pandas.pydata.org/docs/)

