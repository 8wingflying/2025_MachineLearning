# k-nearest-neighbours
## 請以步驟形式說明-Nearest Neighbors (K-NN)
- ChatGPT said:
- 當然！以下是 K-Nearest Neighbors (K-NN) 算法的基本步驟：
- 步驟 1: 收集和準備數據
  - 數據集：你需要一個帶有標籤（類別）的訓練數據集。這些數據應該是具有多個特徵（例如，圖像的像素值或數據的其他屬性）以及每個數據點對應的類別標籤（例如，疾病或健康、不同物種等）。
  - 測試數據：這些是你希望根據訓練數據來預測的數據點。測試數據通常不帶標籤。
- 步驟 2: 計算距離
  - 對於每個測試數據點，計算它與所有訓練數據點的距離。
  - 最常用的距離計算方法是 歐式距離，但你也可以使用其他距離度量（如曼哈頓距離或餘弦相似度）。
- 步驟 3: 找到 K 個最近鄰居
  - 據計算出的距離，將所有訓練數據點按距離從小到大排序。
  - 選擇距離最近的 K 個鄰居。K 是一個超參數，通常由用戶設定。常見的選擇是 K=3 或 K=5。
- 步驟 4: 預測測試點的類別
  - 查看 K 個最近鄰居的類別標籤（例如，每個鄰居的標籤可能是 0 或 1）。
  - 通常，K-NN 使用 多數投票（Majority Voting）來決定測試點的預測類別。也就是說，測試點會被分配給 K 個鄰居中最常見的類別。
  - 如果有多個類別出現相同的次數，則選擇 K 最接近的鄰居中類別最多的標籤。
- 步驟 5: 返回預測結果
  - 對所有測試數據點重複上述過程，得到每個測試點的預測結果。
  - 如果有多個測試點，則會為每個測試點進行分類。
- 步驟 6: 評估模型的效果
  - 準確率（Accuracy）：可以通過比較預測標籤和真實標籤的匹配度來計算模型的準確率：
  - 你也可以使用其他評估指標（如 精確度、召回率、F1 分數 等）來衡量模型的性能，尤其是在處理不平衡數據集時。
- 簡化版 K-NN 流程： 
  - 準備數據：收集帶標籤的訓練數據和不帶標籤的測試數據。
  - 距離計算：對每個測試點，計算它與所有訓練點的距離。
  - 選擇 K 個鄰居：根據距離排序並選擇 K 個最近的鄰居。
  - 預測：使用 K 個鄰居的標籤進行多數投票，預測測試點的類別。
  - 評估：計算模型的準確度或其他性能指標。
- K-NN 優點：
  - 簡單直觀：K-NN 算法非常簡單且易於理解。
  - 無需訓練：它是一個懶惰學習算法（Lazy Learning），不需要訓練過程，直接使用訓練數據來進行預測。
  - 靈活性：可以用於分類和回歸問題，並且可以使用不同的距離度量來調整模型。
- K-NN 缺點：
  - 計算成本高：對於每一個測試點，需要計算所有訓練點的距離，這在大數據集上可能非常慢。
  - 存儲要求高：需要儲存所有訓練數據，這對內存和存儲空間有較高的要求。
  - 對不平衡數據敏感：在類別不平衡的情況下，K-NN 可能會受到影響，因為大類別可能主導預測結果。
- 設定 K 值得小技巧：
  - 選擇合適的 K 值是 K-NN 算法中的關鍵步驟。
  - 過小的 K 值可能會導致模型過度擬合（過度學習訓練數據的噪音）
  - 過大的 K 值則可能會導致模型過於簡單，無法捕捉到數據中的重要細節。
  - 通常，通過交叉驗證（Cross-Validation）來選擇最佳的 K 值。
- 總結：
  - 計算距離：對每個測試點，計算與所有訓練點的距離。
  - 選擇 K 個鄰居：選擇距離最小的 K 個訓練點。
  - 多數投票：根據 K 個鄰居的標籤進行預測。
  - 評估模型：使用準確度或其他評價指標評估模型效果。

## 請用python程式寫一個簡單的k-nearest-neighbours
```python
import numpy as np
from collections import Counter

# 計算兩個點之間的歐式距離
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# K-NN 類別
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """ 訓練模型，保存訓練數據 """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """ 預測測試數據 """
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        """ 預測單個樣本的類別 """
        # 計算所有訓練點與測試點的距離
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 對距離進行排序並選擇 k 個最近的鄰居
        k_indices = np.argsort(distances)[:self.k]
        
        # 獲取 k 個鄰居的標籤
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 計算最常見的標籤
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# 測試數據
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加載 Iiris 數據集
iris = load_iris()
X = iris.data
y = iris.target

# 將數據集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化 K-NN 模型並訓練
knn = KNearestNeighbors(k=3)
knn.fit(X_train, y_train)

# 預測測試集
y_pred = knn.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
```
主要步驟：

KNearestNeighbors 類別：

fit：用來訓練模型，將訓練數據存儲在模型中。

predict：給定測試數據，返回每個測試點的預測結果。

_predict：這是用來計算單個測試點的預測結果，首先計算測試點與所有訓練點的距離，然後選擇最近的 k 個點，並根據最多的類別進行預測。

計算歐式距離：

使用 np.linalg.norm 計算歐式距離，這是 K-NN 中衡量相似度的主要方式。

分類：

利用 Counter 類來統計 k 個鄰居中出現最多的類別，這就是最常見的類別，作為預測結果。

測試與評估：

使用 Scikit-learn 的 train_test_split 將數據集分為訓練集和測試集。

訓練後，預測測試集並計算準確度（accuracy_score）。

測試結果：

這段程式碼使用了著名的 Iris 數據集，並且使用 3 個鄰居進行預測。執行後會輸出模型在測試集上的準確度，類似以下結果：

Accuracy: 95.56%
```
