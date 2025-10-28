# 📘 TF-IDF 教學文件

## 目錄
1. [什麼是 TF-IDF](#什麼是-tf-idf)
2. [數學定義與公式](#數學定義與公式)
3. [直覺理解](#直覺理解)
4. [Python 實作（scikit-learn）](#python-實作scikit-learn)
5. [手動計算範例（純 Python）](#手動計算範例純-python)
6. [常見應用](#常見應用)
7. [延伸技術](#延伸技術)
8. [視覺化範例](#視覺化範例)
9. [參考資料](#參考資料)

---

## 🔹 什麼是 TF-IDF
**TF-IDF（Term Frequency – Inverse Document Frequency）** 是一種衡量詞語在文件集合中重要性的方法，常用於：
- 文件檢索（如搜尋引擎）
- 文本分類（如垃圾郵件分類）
- 特徵提取（如機器學習文本表示）

它的核心思想是：
> 一個詞若在某篇文件中出現頻繁，但在其他文件中很少出現，則該詞對這篇文件具有高權重。

---

## 🔹 數學定義與公式

### (1) Term Frequency (TF)
衡量詞語在文件中出現的頻率。

\[
TF(t, d) = \frac{\text{詞 t 在文件 d 中出現次數}}{\text{文件 d 中所有詞的總數}}
\]

### (2) Inverse Document Frequency (IDF)
衡量詞語在所有文件中出現的普遍性（越普遍 → 越不重要）。

\[
IDF(t) = \log \frac{N}{1 + df(t)}
\]

其中：
- \( N \)：文件總數  
- \( df(t) \)：包含詞 \( t \) 的文件數量  
- 加上 1 防止分母為零

### (3) TF-IDF 加權
\[
TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)
\]

---

## 🔹 直覺理解

| 詞語 | 在文件A出現次數 | 在所有文件中出現次數 | 解釋 |
|------|----------------|--------------------|------|
| data | 10             | 100                | 常見字，權重低 |
| mining | 3             | 10                 | 在少數文件出現，權重高 |
| the | 15              | 500                | 停用詞，權重幾乎為0 |

TF 代表「局部重要性」，IDF 代表「全域稀有性」——兩者結合後可強化關鍵字。

---

## 🔹 Python 實作（scikit-learn）

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 範例文件集
docs = [
    "I love machine learning and data mining",
    "Data mining is a key technique in machine learning",
    "Deep learning drives AI innovation"
]

# 建立 TF-IDF 向量化器
vectorizer = TfidfVectorizer(stop_words='english')

# 計算 TF-IDF 矩陣
tfidf_matrix = vectorizer.fit_transform(docs)

# 取得詞彙列表
words = vectorizer.get_feature_names_out()

# 顯示結果
import pandas as pd
df = pd.DataFrame(tfidf_matrix.toarray(), columns=words)
print(df.round(3))
```

📊 **輸出示例：**
|     | ai | data | deep | drives | key | learning | love | machine | mining | technique |
|-----|----|------|------|--------|-----|-----------|------|----------|---------|------------|
|文檔1|0.0|0.447|0.0|0.0|0.0|0.447|0.547|0.447|0.447|0.0|
|文檔2|0.0|0.333|0.0|0.0|0.516|0.333|0.0|0.333|0.516|0.516|
|文檔3|0.577|0.0|0.577|0.577|0.0|0.0|0.0|0.0|0.0|0.0|

---

## 🔹 手動計算範例（純 Python）

```python
import math
from collections import Counter

docs = [
    "data science is fun",
    "machine learning uses data science",
    "deep learning and data"
]

# 計算 TF
tf_list = []
for doc in docs:
    words = doc.split()
    count = Counter(words)
    total = len(words)
    tf = {w: count[w] / total for w in count}
    tf_list.append(tf)

# 計算 DF
df = Counter()
for tf in tf_list:
    for term in tf:
        df[term] += 1

# 計算 IDF
N = len(docs)
idf = {term: math.log(N / (1 + df[term])) for term in df}

# 計算 TF-IDF
tfidf = []
for tf in tf_list:
    tfidf_doc = {term: tf[term] * idf[term] for term in tf}
    tfidf.append(tfidf_doc)

print(tfidf)
```

---

## 🔹 常見應用
| 應用領域 | 說明 |
|-----------|------|
| 🔍 搜尋引擎排序 | 比較查詢詞與文件的 TF-IDF 向量相似度（例如 cosine similarity） |
| 📧 垃圾郵件分類 | 轉換郵件內容為 TF-IDF 特徵後送入機器學習模型 |
| 🧠 主題建模 | 與 LDA、NMF 結合用於關鍵詞萃取 |
| 🗂️ 文件聚類 | 使用 TF-IDF 向量進行 K-Means 聚類 |

---

## 🔹 延伸技術
| 技術名稱 | 說明 |
|-----------|------|
| **n-gram TF-IDF** | 考慮詞組（如「machine learning」）的權重 |
| **TF-IDF + SVD (LSA)** | 進行降維，提取語義主成分 |
| **TF-IDF + Word2Vec/BERT** | 混合嵌入方法，提升語義理解能力 |

---

## 🔹 視覺化範例

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df, cmap="YlGnBu")
plt.title("TF-IDF Heatmap")
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.show()
```

這段程式碼可將 TF-IDF 權重以熱圖方式視覺化，幫助了解每個詞在不同文件的重要程度。

---

## 🔹 參考資料
- [scikit-learn TfidfVectorizer 官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Stanford NLP 課程筆記](https://web.stanford.edu/class/cs124/)
- Manning, Raghavan, Schütze. *Introduction to Information Retrieval* (Cambridge University Press)

