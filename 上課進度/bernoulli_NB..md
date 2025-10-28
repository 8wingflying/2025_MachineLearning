# Bernoulli Naive Bayes 伯努利貝氏分類

---

## Part 1 | Concept & Intuition 概念與直覺

- **Bernoulli Naive Bayes (BNB)** 適用於 **二元特徵資料**（例如文字是否出現、事件是否發生、0/1 特徵）。
- 它假設每個特徵都遵循 **伯努利分佈 (Bernoulli Distribution)**，即只有「出現 (1)」或「未出現 (0)」兩種情況。

### ✅ 適用情境
- 文字分類（例如垃圾郵件過濾）  
- 特徵是布林值（例如是否點擊、是否開啟郵件）  
- Binary encoding 特徵

---

## Part 2 | Mathematical Foundation 數學基礎

對於每一類別 \( C_k \)，Bernoulli Naive Bayes 的條件機率為：

$$
P(X|C_k) = \prod_{i=1}^{n} P(x_i|C_k)^{x_i} \cdot (1 - P(x_i|C_k))^{(1 - x_i)}
$$

分類規則：

$$
\hat{C} = \arg\max_{C_k} P(C_k) \prod_{i=1}^{n} P(x_i|C_k)^{x_i} (1 - P(x_i|C_k))^{1 - x_i}
$$

---

## Part 3 | Python Implementation (垃圾郵件 Spam vs Ham)

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
texts = [
    "Win a free iPhone now", "Lowest price guaranteed",
    "Hey, are we meeting tomorrow?", "Your invoice is attached",
    "Congratulations, you won lottery", "Let's grab lunch today"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = ham

# Binary bag-of-words representation
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)
y = labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train BernoulliNB model
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

# Predict
y_pred = bnb.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## Part 4 | Feature Log Probabilities 特徵演化比量

```python
import numpy as np

feature_names = np.array(vectorizer.get_feature_names_out())
spam_class_index = list(bnb.classes_).index(1)
top_features = np.argsort(bnb.feature_log_prob_[spam_class_index])[-10:]

print("Spam keywords:")
print(feature_names[top_features])
```

---

## Part 5 | Comparison with Other Naive Bayes Models 比較表

| 模型類型 | 特徵型態 | 分佈假設 | 常見應用 |
|-----------|-----------|-----------|-----------|
| **GaussianNB** | 連續 | 正態分佈 | 數值型特徵 |
| **MultinomialNB** | 計數 | 多項分佈 | 文字分類（字頻） |
| **BernoulliNB** | 二元 | 伯努利分佈 | 文字是否出現 |

---

## Part 6 | Performance Comparison: BernoulliNB vs MultinomialNB

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Prepare vectorizer for MultinomialNB (use frequency counts)
vectorizer_count = CountVectorizer(binary=False)
X_count = vectorizer_count.fit_transform(texts)

bnb_scores = cross_val_score(BernoulliNB(), X, y, cv=3)
mnb_scores = cross_val_score(MultinomialNB(), X_count, y, cv=3)

print(f"BernoulliNB Accuracy: {bnb_scores.mean():.3f}")
print(f"MultinomialNB Accuracy: {mnb_scores.mean():.3f}")
```

**分析結果：**
- 若文本短、詞頻資訊不足 → BernoulliNB 表現較好。
- 若文本長且字詞頻率差異大 → MultinomialNB 通常更準確。

---

## Part 7 | Pros & Cons 優缺點分析

| 優點 | 缺點 |
|------|------|
| 適合布林特徵與稀疏矩陣 | 非二元特徵需轉換 |
| 計算快速、訓練時間短 | 假設特徵獨立性過新 |
| 短文本分類效果良好 | 無法利用字頻資訊 |

---

## Part 8 | Practical Tips 實務建議

1. 對 Binary 特徵最適用，如是否出現、是否點擊、是否存在關鍵詞。  
2. 在短文本上比 MultinomialNB 更有效，而在長文本則相反。  
3. 組合 TF-IDF 向量化或特徵選擇，可顯著提升效能。  
4. 可作為 NLP baseline 用來測試基礎模型效能。

---

✨ **End of Tutorial** — `Bernoulli_NB_Tutorial.md`
