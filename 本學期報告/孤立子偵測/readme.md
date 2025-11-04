# 報告2:
- 名稱:孤立子偵測與不平衡學習演算法分析與報告
- 檔案名稱:A888168_孤立子偵測與不平衡學習演算法分析與報告_20251007

# 測試資料集
#### 測試資料集1: [Credit Card Fraud Detection](kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 真實歐洲信用卡交易資料，284,807 筆，其中詐欺樣本僅 492 筆	
- 極端不平衡資料集，適合練習「異常樣本偵測」
#### 測試資料集2: [Paysim Synthetic Financial Dataset](kaggle.com/datasets/ealaxi/paysim1)
- 模擬金融交易資料，交易類型多樣、含異常交易標籤	模擬大型資料異常偵測，支援多維度特徵
#### 測試資料集3: [Data Cleaning Challenge: Outliers](kaggle.com/code/rtatman/data-cleaning-challenge-outliers)
- Kaggle 官方清理挑戰題，包含真實世界報銷資料	適合初學練習離群點視覺化與資料清理流程Outlier

## 額外測試套件
- 自動化異常偵測框架：
  - PyOD（Python Outlier Detection Library）
    - `論文`[PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection](https://www.arxiv.org/abs/2412.12154)
    - https://pyod.readthedocs.io/en/latest/
  - ADTK（Anomaly Detection Toolkit）時間序列

# 參考資料:Anomaly Detection
- https://github.com/dataman-git/Handbook-of-anomaly-detection
- https://medium.com/dataman-in-ai/handbook-of-anomaly-detection-1-introduction-39b799aab933

# 📘 測試演算法 精簡版
- 統計分析
  - z-score法
  - IQR法
- 機器學習
  - KNN
  - Isolation Forest
    - https://medium.com/dataman-in-ai/handbook-of-anomaly-detection-4-isolation-forest-170615222ab8 
  - One-Class SVM
  -  ...

# 📘 測試演算法 詳盡版
---

## 🧮 一、統計方法（Statistical Methods）

| 方法 | 概念 | 優點 | 缺點 | 適用情境 |
|------|------|------|------|-----------|
| **Z-Score (標準差法)** | 利用平均值與標準差，判斷資料距離平均有多遠（通常 \|Z\| > 3 視為離群） | 簡單直覺 | 對非常態分布資料不準 | 常態分布或近似常態的資料 |
| **Modified Z-Score** | 使用中位數與MAD（Median Absolute Deviation）取代平均與標準差 | 對極端值更穩健 | 不適合多維資料 | 小樣本或含極端值的資料 |
| **IQR (四分位距)** | 以Q1與Q3為界，若超出 Q1−1.5×IQR 或 Q3+1.5×IQR 視為離群 | 簡單易用 | 只適用於單變量 | 盒鬚圖法的基礎，適合初步探索 |
| **Grubbs Test** | 假設資料為常態分布，檢驗最極端值是否為離群 | 有統計顯著性依據 | 需常態假設 | 小樣本或單變量資料 |
| **Dixon’s Q Test** | 針對小樣本數據檢測最小或最大值是否離群 | 簡單 | 僅適用少量樣本 | n < 30 的小樣本實驗資料 |

---

## 🤖 二、機器學習方法（Machine Learning Methods）

| 方法 | 概念 | 優點 | 缺點 | 適用情境 |
|------|------|------|------|-----------|
| **Isolation Forest** | 隨機切割特徵空間，離群點更容易被「孤立」 | 高維資料有效、快速 | 需調參（如 contamination） | 多維數值資料、異常偵測 |
| **One-Class SVM** | 學習資料的邊界，偵測落在邊界外的點 | 適用非線性分佈 | 對超參敏感 | 正常資料多、異常少 |
| **LOF (Local Outlier Factor)** | 比較樣本與鄰居密度差異 | 能偵測局部離群 | 計算量大 | 非均勻分佈資料 |
| **DBSCAN** | 聚類過程中自動判定低密度樣本為離群 | 不需預設群數 | 需調整 ε 與 minPts | 含噪聲的群集分析 |
| **KNN-based Outlier Detection** | 計算每個點與K個最近鄰的距離 | 簡單直覺 | 高維下效果差 | 資料維度不高、樣本量中等 |

---

## 🧠 三、深度學習方法（Deep Learning Methods）

| 方法 | 概念 | 優點 | 缺點 | 適用情境 |
|------|------|------|------|-----------|
| **Autoencoder** | 利用自編碼器重建誤差，誤差大者為離群 | 能處理高維資料 | 需訓練模型 | 時序或影像異常檢測 |
| **Variational Autoencoder (VAE)** | 在隱變量空間中學習資料分佈 | 理論嚴謹 | 複雜、需GPU | 高維異常模式分析 |
| **LSTM Autoencoder** | 適用時間序列資料的重建誤差分析 | 可偵測時序異常 | 訓練時間長 | IoT感測、金融異常行為偵測 |
| **GAN-based Outlier Detection** | 使用生成對抗網路區分正常與異常 | 能處理複雜資料型態 | 訓練不穩定 | 影像或高維異常偵測 |

---


