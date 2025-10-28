# Bayes Classification Tutorial 贝氏定理與分類教學

---

## Part 1 | Bayes Theorem 贝氏定理

### 🧬 Definition 定義
Bayes 定理用於說明在觀察到資料 X 後，某個假設 H 為真的機率。

$$
P(H|X) = \frac{P(X|H) \cdot P(H)}{P(X)}
$$

其中：
- \( P(H|X) \)：後驗機率 (Posterior Probability)
- \( P(H) \)：先驗機率 (Prior Probability)
- \( P(X|H) \)：似然度 (Likelihood)
- \( P(X) \)：資料的總體機率

### 🧮 Intuition 直覺意義
Bayes 定理可以用於更新我們對一個假設的信心程度：
> 「在新資料出現後，我們對某假設為真的信心會怎麼改變？」

---

## Part 2 | Bayesian Classification 贝氏分類

試著依據特徵 X = (x1, x2, ..., xn)，判斷所屬類別 Ck。

$$
\hat{C} = \arg\max_{C_k} P(C_k|X)
$$

使用 Bayes 定理：

$$
P(C_k|X) = \frac{P(X|C_k) \cdot P(C_k)}{P(X)}
$$

由於 \( P(X) \) 相同，可簡化為：

$$
\hat{C} = \arg\max_{C_k} P(X|C_k) \cdot P(C_k)
$$

---

## Part 3 | Naive Bayes Classifier 簡單贝氏分類器

### 🔍 Independence Assumption 條件獨立假設

$$
P(X|C_k) = \prod_{i=1}^{n} P(x_i|C_k)
$$

以此可以大幅降低運算複雜度，是 Naive Bayes 的核心特點。

### 🔢 Common Types 常見類型

| 類型 | 特徵資料型態 | 條件分佈 | 實例 |
|------|----------------|--------------------|------|
| Gaussian NB | 連續 | 正態分佈 | Iris / Wine |
| Multinomial NB | 雜合計數 | 多項分佈 | 文件分類 |
| Bernoulli NB | 0/1 | 伯努利分佈 | 文字出現與否 |

---

## Part 4 | Python Implementation (Gaussian Naive Bayes)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### 🌐 Decision Boundary Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=4)

model = GaussianNB().fit(X, y)

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Gaussian Naive Bayes Decision Boundary")
plt.show()
```

---

## Part 5 | Extended Chapter: Multinomial Naive Bayes 文字分類

### 🔹 Example: IMDB Sentiment Analysis

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
categories = ['rec.autos', 'rec.sport.baseball', 'talk.politics.mideast', 'sci.space']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# Train model
model = MultinomialNB()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
print("Accuracy:", accuracy_score(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred, target_names=categories))
```

### 🔹 View Top Keywords per Class

```python
import numpy as np

feature_names = np.array(vectorizer.get_feature_names_out())
for i, category in enumerate(categories):
    top10 = np.argsort(model.coef_[i])[-10:]
    print(f"\nTop keywords for class '{category}':")
    print(feature_names[top10])
```

---

## Part 6 | Pros & Cons 優缺點分析

| 優點 | 缺點 |
|------|------|
| 計算效率高，適合大量資料 | 條件獨立假設過新 |
| 模型簡單、可解釋性高 | 特徵相關性高時準確率降低 |
| 適用於文字分類、垃圾郵件檢測 | 連續資料需假設分佈 |

---

## Part 7 | Applications 應用範例

| 領域 | 應用範例 |
|-----------|----------------|
| 垃圾郵件過濾 | 判斷郵件是否為垃圾 |
| 文件分類 | 自動新聞分類 |
| 情感分析 | 判斷影評正負面 |
| 醫學诊斷 | 根據症狀預測疾病類型 |

---

## Part 8 | Practical Notes 實務建議

1. 如果特徵不獨立，可考慮 **ComplementNB** 或混合型模型。  
2. 可與 **TF-IDF + Feature Selection** 結合以提升準確率。  
3. 在大規模 NLP 資料上，仍然是效率最高的基素分類方法之一。

---

✨ **End of Tutorial** — `Bayes_Classification_Tutorial.md`

