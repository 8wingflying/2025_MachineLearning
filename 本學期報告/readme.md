
# 報告 
- 報告 1[監督式與非監督式學習演算法實戰報告](iris.md)
  - 以iris資料集為資料集(可擴充到其他資料集)
- 報告2:[孤立子偵測與不平衡學習演算法分析與報告](Outliear.md)
  - 機器學習專案 ==> API or web(Streamlit) 
  - 針對三種資料集進行分析
- 報告3:推薦系統實戰
- 報告4:時間序列分析

## 學習成熟度模型
- level ==> 寫一個演算法 outperform all others
- level ==> 讀最新論文 開發一個有效率演算法實作(含數學推導)
- level  ==> 數學推導 ==> OO python class
- level ==> 使用 套件 ==> 參數調校

## 期末報告
- OBS
- 建一個套件(一堆class .py)
  - A888168
    - `__init__`
    - GMM
    - kmeans
    - kmeans++
    - .....
- streamlit 佈署

## 範例
- GNB
```python
import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes_ = None
        self.means_ = {}
        self.vars_ = {}
        self.priors_ = {}

    def _gaussian_log_pdf(self, x, mean, var):
        """計算 Gaussian log likelihood"""
        eps = 1e-9   # 防止除以零
        return -0.5 * np.log(2 * np.pi * var + eps) - ((x - mean) ** 2) / (2 * var + eps)

    def fit(self, X, y):
        """訓練模型，計算每類的平均、變異數、先驗機率"""
        self.classes_ = np.unique(y)

        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = X_c.mean(axis=0)
            self.vars_[c] = X_c.var(axis=0)
            self.priors_[c] = X_c.shape[0] / X.shape[0]

        return self

    def predict(self, X):
        """預測多筆資料"""
        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)

    def _predict_one(self, x):
        """預測單筆資料"""
        posteriors = []

        for c in self.classes_:
            # log P(y=c)
            log_prior = np.log(self.priors_[c])

            # Σ log P(x_i | y=c)
            log_likelihood = np.sum(
                self._gaussian_log_pdf(x, self.means_[c], self.vars_[c])
            )

            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        # 回傳 posterior 最大的類別
        return self.classes_[np.argmax(posteriors)]


```
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import GNB

iris = load_iris()
X = iris.data
y = iris.target

# 分訓練 / 測試
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 建立模型
model = GaussianNaiveBayes()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 顯示準確率
acc = (y_pred == y_test).mean()
print("模型準確率:", acc)

# 顯示部分預測
print("前十筆預測：", y_pred[:10])
print("前十筆真值：", y_test[:10])
```
