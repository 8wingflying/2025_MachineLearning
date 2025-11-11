
prepare a full pandas-ta trading strategy example that loads real stock data from Yahoo Finance and applies multiple indicators for backtesting.
Here's a complete example of a trading strategy using the pandas-ta library that loads real stock data from Yahoo Finance, applies multiple technical indicators, and sets up a basic backtesting framework.
```
📈 Full Trading Strategy with pandas-ta
🛠️ Step 1: Install Required Libraries
bash
pip install pandas-ta yfinance matplotlib
Optional (for candlestick patterns):

bash
pip install TA-Lib
📥 Step 2: Load Stock Data from Yahoo Finance
python
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import matplotlib.pyplot as plt

# Load historical data for a stock (e.g., Apple)
symbol = "AAPL"
df = yf.download(symbol, start="2022-01-01", end="2023-01-01")
📊 Step 3: Apply Technical Indicators
python
# Add RSI
df["RSI"] = ta.rsi(df["Close"], length=14)

# Add MACD
macd = ta.macd(df["Close"])
df = pd.concat([df, macd], axis=1)

# Add Bollinger Bands
bbands = ta.bbands(df["Close"], length=20)
df = pd.concat([df, bbands], axis=1)

# Add 50-day and 200-day SMA
df["SMA50"] = ta.sma(df["Close"], length=50)
df["SMA200"] = ta.sma(df["Close"], length=200)
📐 Step 4: Define a Simple Trading Strategy
python
# Buy when RSI < 30 and MACD histogram turns positive
# Sell when RSI > 70 and MACD histogram turns negative

df["Signal"] = 0
df.loc[(df["RSI"] < 30) & (df["MACDh_1"] > 0), "Signal"] = 1  # Buy
df.loc[(df["RSI"] > 70) & (df["MACDh_1"] < 0), "Signal"] = -1  # Sell
🔁 Step 5: Backtest Strategy
python
df["Position"] = df["Signal"].replace(to_replace=0, method="ffill")
df["Daily Return"] = df["Close"].pct_change()
df["Strategy Return"] = df["Daily Return"] * df["Position"]

# Calculate cumulative returns
df["Cumulative Market Return"] = (1 + df["Daily Return"]).cumprod()
df["Cumulative Strategy Return"] = (1 + df["Strategy Return"]).cumprod()
📉 Step 6: Visualize Performance
python
plt.figure(figsize=(12,6))
plt.plot(df["Cumulative Market Return"], label="Market Return")
plt.plot(df["Cumulative Strategy Return"], label="Strategy Return")
plt.title(f"{symbol} Strategy vs Market")
plt.legend()
plt.show()
✅ Optional Enhancements
Add stop-loss and take-profit logic

Use vectorbt or backtrader for more advanced backtesting

Optimize parameters using grid search or genetic algorithms

Would you like me to help you extend this strategy with candlestick patterns or integrate it with a portfolio of stocks?

Add stop-loss and take-profit logic Use vectorbt or backtrader for more advanced backtesting
Great! Let's enhance your strategy with stop-loss and take-profit logic, and then show how to backtest it using both vectorbt and backtrader — two powerful Python libraries for quantitative trading.

🛡️ Add Stop-Loss and Take-Profit Logic
We'll define:

Stop-loss: Exit if price drops more than 5% from entry.

Take-profit: Exit if price rises more than 10% from entry.

Example Logic in Pandas
python
entry_price = None
df["Trade"] = 0

for i in range(1, len(df)):
    if df["Signal"].iloc[i] == 1 and entry_price is None:
        entry_price = df["Close"].iloc[i]
        df.at[df.index[i], "Trade"] = 1  # Enter trade
    elif entry_price:
        change = (df["Close"].iloc[i] - entry_price) / entry_price
        if change <= -0.05 or change >= 0.10 or df["Signal"].iloc[i] == -1:
            df.at[df.index[i], "Trade"] = -1  # Exit trade
            entry_price = None
⚙️ Advanced Backtesting with vectorbt
Step-by-Step Setup
bash
pip install vectorbt
Strategy with Indicators and SL/TP
python
import vectorbt as vbt

# Load data
price = yf.download("AAPL", start="2022-01-01", end="2023-01-01")["Close"]

# Indicators
rsi = ta.rsi(price, length=14)
macd = ta.macd(price)["MACDh_1"]

# Entry/Exit signals
entries = (rsi < 30) & (macd > 0)
exits = (rsi > 70) | (macd < 0)

# SL/TP settings
sl_pct = 0.05
tp_pct = 0.10

# Run backtest
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    sl_stop=sl_pct,
    tp_stop=tp_pct,
    direction="longonly"
)

pf.plot().show()
🔍 Advanced Backtesting with backtrader
Setup
bash
pip install backtrader yfinance
Strategy Class with SL/TP
python
import backtrader as bt

class RSIMACDStrategy(bt.Strategy):
    params = dict(rsi_period=14, macd1=12, macd2=26, macdsig=9, sl=0.05, tp=0.10)

    def __init__(self):
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)
        self.macd = bt.ind.MACD(
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macdsig
        )
        self.entry_price = None

    def next(self):
        if not self.position:
            if self.rsi < 30 and self.macd.macd > self.macd.signal:
                self.buy()
                self.entry_price = self.data.close[0]
        else:
            change = (self.data.close[0] - self.entry_price) / self.entry_price
            if change <= -self.p.sl or change >= self.p.tp or (self.rsi > 70 and self.macd.macd < self.macd.signal):
                self.close()

# Run engine
cerebro = bt.Cerebro()
data = bt.feeds.YahooFinanceData(dataname="AAPL", fromdate=pd.Timestamp("2022-01-01"), todate=pd.Timestamp("2023-01-01"))
cerebro.adddata(data)
cerebro.addstrategy(RSIMACDStrategy)
cerebro.run()
cerebro.plot()
Would you like to compare performance metrics between vectorbt and backtrader, or explore portfolio-level strategies across multiple tickers?

portfolio-level strategies across multiple tickers
To build a portfolio-level trading strategy across multiple tickers using pandas-ta, vectorbt, or backtrader, you’ll want to:

Load and align data for multiple assets

Apply indicators and generate signals per asset

Allocate capital and simulate trades across the portfolio

Evaluate performance at both asset and portfolio level

Here’s how to do it with both vectorbt and backtrader:

🧠 Option 1: Portfolio Strategy with vectorbt
✅ Advantages
Vectorized, fast, and memory-efficient

Easy to scale across many tickers

Built-in portfolio analytics

📦 Step-by-Step Example
python
import vectorbt as vbt
import yfinance as yf

# Define tickers
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

# Download price data
price = yf.download(tickers, start="2022-01-01", end="2023-01-01")["Close"]

# Compute RSI and MACD
rsi = vbt.RSI.run(price, window=14).rsi
macd = vbt.MACD.run(price)

# Entry/exit logic per asset
entries = (rsi < 30) & (macd.hist > 0)
exits = (rsi > 70) | (macd.hist < 0)

# Backtest portfolio
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    size=0.2,  # 20% allocation per asset
    direction="longonly",
    sl_stop=0.05,
    tp_stop=0.10
)

# Plot performance
portfolio.total_return().vbt.barplot(title="Total Return per Asset")
portfolio.plot()
🧠 Option 2: Portfolio Strategy with backtrader
✅ Advantages
More flexible for complex logic

Supports cash management, commissions, slippage

Great for live trading integration

🧰 Setup
python
import backtrader as bt
import yfinance as yf
import pandas as pd

class MultiAssetStrategy(bt.Strategy):
    params = dict(sl=0.05, tp=0.10)

    def __init__(self):
        self.order_refs = {}
        self.entry_prices = {}

    def next(self):
        for d in self.datas:
            pos = self.getposition(d)
            if not pos:
                if d.rsi[0] < 30 and d.macd.macd[0] > d.macd.signal[0]:
                    self.order_refs[d._name] = self.buy(data=d)
                    self.entry_prices[d._name] = d.close[0]
            else:
                change = (d.close[0] - self.entry_prices[d._name]) / self.entry_prices[d._name]
                if change <= -self.p.sl or change >= self.p.tp or (d.rsi[0] > 70 and d.macd.macd[0] < d.macd.signal[0]):
                    self.close(data=d)

# Load data and indicators
tickers = ["AAPL", "MSFT", "GOOGL"]
cerebro = bt.Cerebro()

for ticker in tickers:
    df = yf.download(ticker, start="2022-01-01", end="2023-01-01")
    df["rsi"] = bt.ind.RSI(bt.feeds.PandasData(dataname=df), period=14)
    df["macd"] = bt.ind.MACD(bt.feeds.PandasData(dataname=df))
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data, name=ticker)

cerebro.addstrategy(MultiAssetStrategy)
cerebro.run()
cerebro.plot()
📊 Portfolio Metrics to Track
Total return and volatility

Sharpe and Sortino ratios

Max drawdown

Correlation between assets

Capital allocation over time

Would you like to add dynamic rebalancing, risk parity weighting, or sector-based filtering to this portfolio strategy?
```
