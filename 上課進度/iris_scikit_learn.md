# 📚 **Scikit-learn** 
> **Scikit-learn** 是 Python 機器學習的核心套件，  
> 涵蓋 **分類、迴歸、分群、降維、前處理、特徵工程與模型評估**。  
>  
> 它以 **一致的 API 設計、豐富的模型支援與 sklearn.pipeline 整合性**  
> 成為機器學習教育與產業應用的標準工具。  

---

# 🤖 Scikit-learn 各項功能與常用函數總覽表  


---

## 一、Scikit-learn 概要（Overview）

| 模組名稱 | 中文說明 | 功能重點 |
|-----------|------------|-----------|
| `sklearn.datasets` | 內建資料集 | 常用測試資料，如 iris、digits |
| `sklearn.preprocessing` | 資料前處理 | 標準化、正規化、編碼 |
| `sklearn.model_selection` | 模型選擇 | 資料分割、交叉驗證、網格搜尋 |
| `sklearn.linear_model` | 線性模型 | 迴歸與分類（Logistic、Ridge、Lasso） |
| `sklearn.svm` | 支援向量機 | 分類與迴歸 |
| `sklearn.tree` | 決策樹 | 分類與迴歸樹模型 |
| `sklearn.ensemble` | 集成學習 | RandomForest、GradientBoosting、Bagging |
| `sklearn.cluster` | 分群演算法 | KMeans、DBSCAN、Agglomerative |
| `sklearn.neighbors` | 近鄰方法 | KNN 分類、回歸、距離計算 |
| `sklearn.naive_bayes` | 樸素貝氏分類 | Gaussian、Multinomial、Bernoulli |
| `sklearn.decomposition` | 降維方法 | PCA、ICA、NMF |
| `sklearn.feature_selection` | 特徵選取 | 過濾法、包裝法、嵌入法 |
| `sklearn.metrics` | 評估指標 | 混淆矩陣、精確率、ROC、F1 |
| `sklearn.pipeline` | 管線化 | 串聯前處理與模型 |
| `sklearn.utils` | 工具模組 | 隨機、驗證、shuffle、計時等 |

---

## 二、資料集模組（`sklearn.datasets`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 載入內建資料集 | 常用標準資料 | `load_iris()`, `load_digits()`, `load_wine()` | 回傳 dict-like 結構 |
| 載入外部資料集 | 從 URL 載入資料 | `fetch_openml('mnist_784')` | — |
| 建立模擬資料 | 生成合成資料集 | `make_classification()`, `make_blobs()`, `make_regression()` | 可自訂分佈與維度 |

---

## 三、資料前處理（`sklearn.preprocessing`）

| 功能類別 | 功能說明 | 常用函數 / 類別 | 備註 |
|------------|------------|------------------|------|
| 標準化 | 均值=0, 方差=1 | `StandardScaler()` | 常用於 SVM、PCA |
| 正規化 | 向量長度為 1 | `Normalizer()` | 適用於距離模型 |
| 最小-最大縮放 | [0,1] 範圍 | `MinMaxScaler()` | — |
| 編碼轉換 | 類別資料編碼 | `LabelEncoder()`, `OneHotEncoder()` | — |
| 缺值處理 | 補缺失值 | `SimpleImputer(strategy='mean')` | — |
| 多項式特徵 | 建立高次項 | `PolynomialFeatures(degree=2)` | 非線性模型常用 |
| 二值化 | 將特徵轉為 0/1 | `Binarizer(threshold=0.5)` | — |

---

## 四、資料分割與模型選擇（`sklearn.model_selection`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 訓練/測試分割 | 拆分資料集 | `train_test_split(X, y, test_size=0.2)` | 隨機抽樣 |
| 交叉驗證 | 模型泛化驗證 | `cross_val_score(model, X, y, cv=5)` | — |
| 分層抽樣 | 保持類別比例 | `StratifiedKFold(n_splits=5)` | 不平衡資料常用 |
| 網格搜尋 | 自動調參 | `GridSearchCV(model, param_grid, cv=5)` | 尋找最佳參數 |
| 隨機搜尋 | 隨機搜尋最佳參數 | `RandomizedSearchCV(model, param_distributions)` | 加速搜尋 |
| 學習曲線 | 檢視模型表現 | `learning_curve(model, X, y)` | — |

---

## 五、線性模型（`sklearn.linear_model`）

| 模型名稱 | 功能說明 | 常用函數 / 類別 | 備註 |
|------------|------------|------------------|------|
| 線性迴歸 | 最小平方法 | `LinearRegression()` | — |
| 岭迴歸 | L2 正規化 | `Ridge(alpha=1.0)` | 防止 overfitting |
| 套索迴歸 | L1 正規化 | `Lasso(alpha=0.1)` | 特徵選擇 |
| 彈性網 | L1+L2 混合 | `ElasticNet(alpha, l1_ratio)` | — |
| 邏輯斯迴歸 | 分類模型 | `LogisticRegression()` | 支援多類別 |
| 貝氏迴歸 | Bayesian Regression | `BayesianRidge()` | 機率式模型 |

---

## 六、支援向量機（`sklearn.svm`）

| 模型名稱 | 功能說明 | 常用函數 / 類別 | 備註 |
|------------|------------|------------------|------|
| SVC | 支援向量分類 | `SVC(kernel='rbf', C=1.0, gamma='scale')` | 最常用分類器 |
| SVR | 支援向量迴歸 | `SVR(kernel='linear')` | — |
| LinearSVC | 線性核快速版本 | `LinearSVC()` | 適合大資料 |
| OneClassSVM | 異常值偵測 | `OneClassSVM()` | — |

---

## 七、樹與集成模型（`tree`, `ensemble`）

| 模型名稱 | 所屬模組 | 功能說明 | 常用函數 / 類別 |
|------------|-----------|------------|------------------|
| 決策樹 | `sklearn.tree` | 分類與迴歸樹 | `DecisionTreeClassifier()`, `DecisionTreeRegressor()` |
| 隨機森林 | `sklearn.ensemble` | 多樹平均 | `RandomForestClassifier()`, `RandomForestRegressor()` |
| 梯度提升 | `sklearn.ensemble` | Boosting | `GradientBoostingClassifier()`, `HistGradientBoostingClassifier()` |
| AdaBoost | `sklearn.ensemble` | 加權集成 | `AdaBoostClassifier()` |
| Bagging | `sklearn.ensemble` | 複製樣本訓練多模型 | `BaggingClassifier()` |
| ExtraTrees | `sklearn.ensemble` | 隨機性更高的森林 | `ExtraTreesClassifier()` |
| Stacking | `sklearn.ensemble` | 模型堆疊集成 | `StackingClassifier()` |

---

## 八、近鄰方法（`sklearn.neighbors`）

| 模型名稱 | 功能說明 | 常用函數 / 類別 | 備註 |
|------------|------------|------------------|------|
| KNN 分類 | K 近鄰分類 | `KNeighborsClassifier(n_neighbors=5)` | 非參數法 |
| KNN 迴歸 | K 近鄰迴歸 | `KNeighborsRegressor()` | — |
| 最近鄰查詢 | 距離搜尋 | `NearestNeighbors()` | 支援 ball_tree / kd_tree |

---

## 九、分群演算法（`sklearn.cluster`）

| 模型名稱 | 功能說明 | 常用函數 / 類別 | 備註 |
|------------|------------|------------------|------|
| K-means | 常用分群法 | `KMeans(n_clusters=3)` | 可視化常見 |
| 層次分群 | 自底向上分群 | `AgglomerativeClustering()` | 可結合樹狀圖 |
| DBSCAN | 密度式分群 | `DBSCAN(eps=0.5, min_samples=5)` | 可偵測噪音點 |
| MeanShift | 均值漂移分群 | `MeanShift()` | — |
| Spectral Clustering | 光譜分群 | `SpectralClustering(n_clusters=3)` | 基於圖論 |

---

## 十、降維方法（`sklearn.decomposition`）

| 方法名稱 | 功能說明 | 常用函數 / 類別 | 備註 |
|------------|------------|------------------|------|
| PCA | 主成分分析 | `PCA(n_components=2)` | 最常用降維方法 |
| Kernel PCA | 非線性降維 | `KernelPCA(kernel='rbf')` | — |
| ICA | 獨立成分分析 | `FastICA()` | 用於信號分離 |
| NMF | 非負矩陣分解 | `NMF(n_components=2)` | 適用於非負資料 |
| TruncatedSVD | 稀疏矩陣降維 | `TruncatedSVD()` | 適用於文本特徵 |

---

## 十一、特徵選取（`sklearn.feature_selection`）

| 功能類別 | 功能說明 | 常用函數 / 類別 | 備註 |
|------------|------------|------------------|------|
| 過濾法 | 根據統計檢定選特徵 | `SelectKBest(score_func=chi2)` | — |
| 逐步選取 | 根據模型重要性 | `RFE(estimator, n_features_to_select)` | 遞迴特徵刪除 |
| L1 正則化選取 | 基於 Lasso 權重 | `SelectFromModel(Lasso())` | 嵌入法 |
| 變異數過濾 | 移除低變異特徵 | `VarianceThreshold(threshold=0.0)` | — |

---

## 十二、評估與指標（`sklearn.metrics`）

| 指標類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 混淆矩陣 | 類別準確性 | `confusion_matrix(y_true, y_pred)` | — |
| 精確率 / 召回率 | 分類表現 | `precision_score()`, `recall_score()`, `f1_score()` | — |
| ROC / AUC | 分類閾值表現 | `roc_curve()`, `roc_auc_score()` | — |
| R² 與 MSE | 迴歸評估 | `r2_score()`, `mean_squared_error()` | — |
| Silhouette | 分群效果 | `silhouette_score(X, labels)` | — |
| Classification Report | 分類摘要 | `classification_report()` | — |

---

## 十三、管線化（`sklearn.pipeline`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 管線化流程 | 串接前處理與模型 | `Pipeline(steps=[('scaler', StandardScaler()), ('model', SVC())])` | — |
| 特徵轉換 | 結合多個轉換器 | `FeatureUnion(transformer_list=[...])` | 並行處理多特徵 |
| 網格搜尋結合 | 與 GridSearchCV 搭配 | `GridSearchCV(Pipeline(...), param_grid)` | 常用於自動化訓練流程 |

---

## 十四、常用輔助工具（`sklearn.utils`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 資料隨機化 | 打亂樣本順序 | `shuffle(X, y)` | — |
| 權重處理 | 樣本加權 | `compute_sample_weight()` | 處理不平衡資料 |
| 檢查模型參數 | 驗證超參數 | `all_estimators()`、`check_X_y()` | — |
| 時間測量 | 模型訓練耗時 | `timeit()`（外部） | — |

---



✅ **典型範例：分類模型管線**

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

param_grid = {'svc__C':[0.1,1,10], 'svc__kernel':['linear','rbf']}
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Accuracy:", grid.score(X_test, y_test))
