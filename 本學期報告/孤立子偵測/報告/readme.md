# 信用卡異常偵測模型開發與建置
- 資料集說明
- EDA 分析
- 特徵工程 ==> 展現有特徵工程處理過 與 沒有特徵工程處理過的有何差異
- 傳統分類演算法實戰
  - 評估指標
  - 需列出那些是異常資料 
- 傳統叢集演算法實戰
  - 評估指標
  - 需列出那些是異常資料
- 無監督異常偵測
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM 
- 降維演算法實戰 
- 半監督演算法實戰 ==> {降維 ==> 分類演算法 ==> cross-validation}
  - UMAP ==> Random Forest
  - UMAP + Random Forest ==> 超參數調教 GridSearchCV + Optuna


## 信用卡偵測模型
- [非監督式學習｜使用 Python](https://www.tenlong.com.tw/products/9789865024062)
  - [Hands-On Unsupervised Learning Using Python](https://learning.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/)
  - https://github.com/aapatel09/handson-unsupervised-learning
- EDA
- 監督式學習 ==>chapter 02　完整的機器學習專案 Precision-Recall Curve
  - Feature Engineering for Time Series Forecasting
  - Model #1: Logistic Regression
  - Model #2: Random Forests
  - Model #3: Gradient Boosting Machine (XGBoost)
  - Model #4: Gradient Boosting Machine (LightGBM)
  - Stacking
- 非監督式學習==> chapter 04　異常偵測
  - Normal PCA Anomaly Detection
  - Sparse PCA Anomaly Detection
  - Kernel PCA Anomaly Detection
  - Gaussian Random Projection Anomaly Detection
  - Sparse Random Projection Anomaly Detection
  - Nonlinear Anomaly Detection
  - Dictionary Learning Anomaly Detection
  - ICA Anomaly Detection
  - Fraud Detection on the Test Set 
- 神經網路==>  chapter 08　實際操作自動編碼器\
  - undercomplete autoencoders 
  - overcomplete autoencoders 
  - sparse autoencoders 
  - denoising autoencoders 
  - variational autoencoders 
- 半監督式學習(Semisupervised Learning)==> chapter 09　半監督式學習
  - baselines
    - a supervised model  ==> light gradient boosting + k-fold cross-validation
    - a unsupervised mode  ==> a sparse two-layer overcomplete autoencoder with a linear activation function
    - Semisupervised Model ==>
- 深度學習  
- NEWS
  - 對抗網路盜刷 聯卡中心擬建立台版信用卡詐欺偵測模型 2024/07/17
  - https://www.cna.com.tw/news/afe/202407170162.aspx
  - 永豐金控ＡＩ科技團隊  https://www.cdns.com.tw/articles/609085
    - 上半年展現高效執行，建置出「智能偵測信用卡盜刷模型」，在零點零零五秒內即可快速預測該筆交易的盜刷機率，並維持百分之九十八的準確率，快速應對潛在刷卡交易風險與詐欺行為
    - 此外，團隊亦將實價登錄資料結合電子地圖資訊，自行開發「不動產自動估價模型」（ＡＶＭ），加速內部鑑價作業流程，協助民眾快速掌握更精確的可貸額度。
  - 【玉山AI實例1】0.1秒揪出信用卡盜刷行為，一年擋下上億元損失
    - https://www.ithome.com.tw/news/146199 
#### 20 種分類演算法
- A. 線性模型 (4 種)
  - Logistic Regression
  - RidgeClassifier
  - SGDClassifier（Log Loss）
  - PassiveAggressiveClassifier
- B. SVM / 核方法 (2 種)
  - Linear SVM（LinearSVC）
  - RBF SVM（SVC(kernel="rbf")）
- C. kNN / 鄰近方法 (1 種)
  - k-Nearest Neighbors (kNN)
- D. 貝氏分類器 (2 種)
  - GaussianNB
  - BernoulliNB
- E. 樹模型 (3 種)
  - Decision Tree
  - ExtraTreeClassifier
  - RandomForest
- F. 集成學習 (5 種)
  - Gradient Boosting
  - AdaBoost
  - XGBoost
  - LightGBM
  - CatBoost
- G. 神經網路 (3 種)
  - MLPClassifier
  - TabNetClassifier（可解釋深度表格模型）
  - Simple PyTorch MLP（手寫）

#### 降維演匴法
- PCA（Principal Component Analysis）
- Kernel PCA（RBF 核）
- Truncated SVD
- ICA（FastICA）
- Gaussian Random Projection
- Isomap
- Locally Linear Embedding（LLE）
- t-SNE
- UMAP
- Autoencoder（深度自編碼器）
- LDA

#### 叢集演算法


#### BAYES Classifier ==> MLE

#### SVD
- https://en.wikipedia.org/wiki/Singular_value_decomposition#Intuitive_interpretations
- https://ccjou.wordpress.com/2009/09/01/%E5%A5%87%E7%95%B0%E5%80%BC%E5%88%86%E8%A7%A3-svd/
- https://www.geeksforgeeks.org/machine-learning/singular-value-decomposition-svd/
- https://en.wikipedia.org/wiki/Singular_value_decomposition
- https://towardsdatascience.com/singular-value-decomposition-svd-demystified-57fc44b802a0/
- https://arindam.cs.illinois.edu/papers/09/anomaly.pdf
- https://www.academia.edu/2691860/Anomaly_detection_A_survey

#### GMM ==> EM


#### [Handbook of Anomaly Detection(參看GITHUB所列的演算法)](https://medium.com/dataman-in-ai/handbook-of-anomaly-detection-1-introduction-39b799aab933)
- https://github.com/dataman-git/Handbook-of-anomaly-detection/tree/main
- (2) HBOS
- (3) ECOD
- (4) Isolation Forest
- (5) PCA
- (6) One-Class SVM
- (7) GMM
- (8) KNN
- (9) Local Outlier Factor (LOF)
- (10) Cluster-Based Local Outlier Factor (CBLOF)
- (11) Autoencoders
- (12) Supervised Learning Primer
- (13) Regularization
- (14) Sampling Techniques for Extremely Imbalanced Data
- (15) Representation Learning for Outlier Detection

## Chatgpt建議
- 用途建議及注意事項

可用於 二分類模型訓練：判斷交易是否為詐欺。

可用於 不平衡資料處理技術實驗，如：過抽樣 (oversampling)、欠抽樣 (undersampling)、SMOTE、成本敏感學習。

可用於 異常偵測研究，將詐欺當作少數類別，採用隔離森林 (Isolation Forest)、AutoEncoder 等非監督方法。

可用於 特徵工程練習：雖然大部分是匿名化特徵，但可試用金額、時間、交易頻率、聚合特徵等衍生變數。

適合用於 模型效能評估：因為真實場景下詐欺比例非常低，需注意常見的「準確率（Accuracy）看似高」但實際沒抓到詐欺的陷阱。
arXiv

注意事項

因為特徵已被 PCA、匿名化，解釋性 (interpretability) 相對較低。若你的研究需要可解釋特徵（例如具體欄位意義、交易類型、商戶資訊等），可能需要其他資料集。

資料集時間範圍僅為兩天，且為歐洲地區某卡片機構，樣本可能未涵蓋所有詐欺型態、地域或長期行為變化。模型若應用於其他地區或時期需慎重。

雖然適合用於學術實驗，但在真正上線（實務銀行／支付機構）時，可能需更多樣本、更長期間、更豐富特徵（如交易地點、商戶類別、卡片持有人行為特徵、社交網絡關係等）。

因為類別不平衡極端，單純使用 Accuracy 可能誤導。建議優先考慮 Recall（召回率）、Precision（精確率）、F1-score 及成本敏感指標。
Medium

若進行交叉驗證或時間序列分割，務必保持「以時間為基準」或「避免資料洩漏 (data leakage)」的設計，因為交易時間 Time 欄位顯示交易先後順序。

