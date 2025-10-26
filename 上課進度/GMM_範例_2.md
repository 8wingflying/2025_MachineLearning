
## 範例程式:高斯混合模型_異常偵測
- 使用高斯混合模型 (Gaussian Mixture Model, GMM) 來偵測異常點
```python

from sklearn.mixture import GaussianMixture
import numpy as np

class GMMAnomalyDetector(GaussianMixture):
    def __init__(self, n_components=2, covariance_type='full', tol=1e-3, outlier_fraction=0.05):
        super().__init__(n_components=n_components, covariance_type=covariance_type, tol=tol)
        self.outlier_fraction = outlier_fraction

    def predict(self, X):
        # Compute likelihood scores
        scores = self.score_samples(X)
        # Determine threshold based on outlier fraction
        threshold = np.percentile(scores, 100 * self.outlier_fraction)
        # Classify anomalies
        return (scores < threshold).astype(int)

# Example usage
X = np.random.rand(100, 2) # Replace with your dataset
gmm_detector = GMMAnomalyDetector(n_components=2, outlier_fraction=0.05)
gmm_detector.fit(X)
anomalies = gmm_detector.predict(X)
print("Anomalies:", anomalies)
```
```
GMMAnomalyDetector 類別:
這個類別繼承自 GaussianMixture，表示它具有標準 GMM 的所有功能，並額外增加了異常點偵測的功能。

__init__(self, n_components=2, covariance_type='full', tol=1e-3, outlier_fraction=0.05):
這是類別的建構子。
它接受幾個參數：
n_components: 高斯混合模型中的高斯分量數量 (預設為 2)。
covariance_type: 使用的共變異數矩陣類型 ('full', 'tied', 'diag', 或 'spherical')。'full' 是預設值。
tol: 收斂閾值 (預設為 1e-3)。
outlier_fraction: 資料集中預期的異常點比例 (預設為 0.05，表示 5%)。這是異常點偵測的關鍵參數。

super().__init__(...): 呼叫父類別 (GaussianMixture) 的建構子，使用指定的參數初始化 GMM。
self.outlier_fraction = outlier_fraction: 將異常點比例儲存為類別的實例變數。

predict(self, X):這個方法接受輸入資料 X，並返回一個陣列，指示每個資料點是否為異常點。
scores = self.score_samples(X):
計算每個資料點在擬合的 GMM 下的對數可能性分數 (log-likelihood scores)。分數越低的資料點被認為可能性越小，因此越有可能是異常點。

threshold = np.percentile(scores, 100 * self.outlier_fraction): 根據對數可能性分數和指定的 outlier_fraction 計算一個閾值。
np.percentile 函式用於找到低於特定百分比資料的值。通過使用 100 * self.outlier_fraction，我們找到對應於最低百分比分數的閾值 (即最異常的分數)。

return (scores < threshold).astype(int): 將每個資料點的分數與計算出的閾值進行比較。
如果資料點的分數低於閾值，則被分類為異常點 (True)，否則被視為正常點 (False)。
.astype(int) 將這些布林值轉換為整數 (異常點為 1，正常點為 0)。
```
```
範例使用:
X = np.random.rand(100, 2): 使用隨機數建立一個包含 100 個資料點和 2 個特徵的樣本資料集 X。您需要將其替換為您的實際資料集。
gmm_detector = GMMAnomalyDetector(n_components=2, outlier_fraction=0.05): 建立一個 GMMAnomalyDetector 的實例，設定 2 個分量和預期異常點比例為 5%。
gmm_detector.fit(X): 將 GMM 模型擬合到樣本資料 X。這一步驟會學習最佳描述資料分布的高斯分量的參數。
anomalies = gmm_detector.predict(X): 使用訓練好的 gmm_detector 預測 X 中的每個資料點是否為異常點。
print("Anomalies:", anomalies): 印出結果陣列，其中 1 表示異常點，0 表示正常點。

總之，這段程式碼通過在您的資料上訓練一個 GMM 來建模其正常分布。
然後，它使用資料點在這個模型下的可能性來識別那些可能性顯著較低的點，並根據指定的異常點比例將它們分類為異常點。
```
