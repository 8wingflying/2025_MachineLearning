
## 範例程式:高斯混合模型_異常偵測
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

```
