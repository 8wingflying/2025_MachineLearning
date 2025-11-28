# Kaggle分析
- 資料集:[MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
- [MovieLens - Collaborative Filtering & SVD + gradio](https://www.kaggle.com/code/fatmaamrabdurrahman/movielens-collaborative-filtering-svd-gradio)
- [Movie Reccomender System with Matrix Factorisation](https://www.kaggle.com/code/abhiramrayadurgam/movie-reccomender-system-with-matrix-factorisation)
- [FROM ZERO TO HERO Implamentating: SVD++, KNN, NCF](https://www.kaggle.com/code/bananalord111/from-zero-to-hero-implamentating-svd-knn-ncf)
- [Boltzman_Machine_Movie_Recommendation](https://www.kaggle.com/code/anlmehmetuyar/boltzman-machine-movie-recommendation)
- [Comparison for Deep Recommender Systems](https://www.kaggle.com/code/hakanerdem/comparison-for-deep-recommender-systems)
- [Deep Learning - AutoEncoder](https://www.kaggle.com/code/sethmackie/deep-learning-autoencoder)

# 推薦系統報告主題
- 推薦系統
- 推薦系統類型
- 實戰資料集:[MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
  - Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies
  - https://grouplens.org/datasets/movielens/100k/
- 資料分析==> EDA
  - [A Detailed Explanation of Keras Embedding Layer](https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer) 
- 機器學習
- 關聯規則分析 Market basket analysis (association rule mining)
  - Ch2@BOOK2 Python推薦系統實戰：基於深度學習、NLP和圖算法的應用型推薦系統
  - https://learning.oreilly.com/library/view/applied-recommender-systems/9781484289549/
  - https://github.com/Apress/applied-recommender-systems-python. 
- 協同過濾分析Collaborative Filtering
  - https://github.com/sudekacar/netflix-recommender/tree/main
    - Title-based Content Filtering ==> Recommend similar movies based on their titles using TF-IDF Vectorization.
    - Collaborative Filtering ==> Recommend movies by finding similar users using matrix factorization techniques. 
  - Singular Value Decomposition (SVD) for collaborative filtering
    - https://github.com/Chhaviroy/movie-recommendation-system-svd  
- 深度學習(Deep Learning)
  - 受限玻爾茲曼機(restricted Boltzmann machine, RBM)`電影推薦系統
    - [非監督式學習｜使用 Python| Ankur A. Patel 著 ch 10](https://www.tenlong.com.tw/products/9789865024062?list_name=srh) 
      - Hands-On Unsupervised Learning Using Python
      - https://github.com/aapatel09/handson-unsupervised-learning
  - NLP-based
    - LDA
    - word2vec
    - item2vec 
- LLM


#### 資料分析
- https://www.kaggle.com/code/yoghurtpatil/movielens-100k-data-analysis
- 1) 使用者的職業中，前3名的電影是什麼？
- 2) 每個電影類別中，前3名的電影是什麼？
- 3) 根據職業和電影類別，前3名是什麼？
- 4) 使用者的每個年齡群組中，前3名的電影是什麼？
- 5) 夏季（五月到七月）發行的前3個電影類別是什麼？
- 6) 對於每個電影類別，前2個共同出現的電影類別是什麼？
- 7) 對於每個使用者，我們能找到另一個有相似偏好嗎？


#### BOOKS
- [實用推薦系統](https://www.tenlong.com.tw/products/9787121420788?list_name=srh)
  - Practical Recommender Systems
- [推薦系統：原理與實踐|Charu C. Aggarwal]()
  - Recommender Systems: The Textbook 
- [統計推薦系統|（美）迪帕克·K.阿加瓦爾（Deepak K. Agarwal）](https://www.tenlong.com.tw/products/9787111635734?list_name=sp)
  - Statistical Methods for Recommender Systems 
- BOOK2[Python推薦系統實戰：基於深度學習、NLP和圖算法的應用型推薦系統](https://www.tenlong.com.tw/products/9787302657408?list_name=lv)
  - [Applied Recommender Systems with Python: Build Recommender Systems with Deep Learning, Nlp and Graph-Based Techniques](https://learning.oreilly.com/library/view/applied-recommender-systems/9781484289549/)
  - https://github.com/Apress/applied-recommender-systems-python
  - 第2章 超市購物車分析（關聯規則挖掘）
  - 第3章 內容過濾
  - 第4章 協同過濾
  - 第5章 使用矩陣分解、奇異值分解和共聚類的協同過濾
  - 第7章 聚類cluster
  - 第8章 分類 ==> 邏輯回歸  決策樹  隨機森林 KNN
  - 第9章 基於深度學習的推薦系統  ==> 神經協同過濾（NCF）
    - Neural Collaborative Filtering (NeuralCF)
    - [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
    - https://bbs.huaweicloud.com/blogs/328532
    - https://www.modb.pro/db/178133
    - https://zhuanlan.zhihu.com/p/452679225
    - 評價指標NDCG（CG、DCG、IDCG）
    - CG（Cumulative Gain）累計收益
    - DCG（Discounted cumulative gain）折扣累計收益
    - NDCG（Normalize DCG）歸一化折扣累計收益
  - 第10章 基於圖
  - 第11章 新興領域和新技術
- [大模型推薦系統實戰 從預訓練到智能代理部署|劉璐 張玉君](https://www.tenlong.com.tw/products/9787115675569)
- [大模型智能推薦系統：技術解析與開發實踐|梁誌遠、韓曉晨](https://www.tenlong.com.tw/products/9787302685654?list_name=srh)
- 已買[推薦系統核心技術與實踐|遊雪琪、劉建濤](https://www.tenlong.com.tw/products/9787302681946)

## 論文研讀
- `202504REVIEW`[A Survey of Foundation Model-Powered Recommender Systems: From Feature-Based, Generative to Agentic Paradigms](https://arxiv.org/abs/2504.16420)
- 202305[TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation](https://arxiv.org/abs/2305.00447)
- MIND： 一個大規模英文新聞推薦的公開數據集
- MIND News Recommendation Competition 
- [MIND: MIcrosoft News Dataset](https://msnews.github.io/)
- 1百萬使用者和16萬篇新聞（類別、標題、摘要、全文和抽取的實體）
- 新聞推薦更偏向於NLP任務，而不是傳統的推薦系統（協同過濾、矩陣分解等手段得到id embedding表示）
- 2020[MIND: ALarge-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)

## 深度學習推薦系統的業界經典案例
- [深度學習推薦系統 2.0|王喆](https://www.tenlong.com.tw/products/9787121497469?list_name=srh)
- YouTube深度學習視頻推薦系統
  - https://www.youtube.com/howyoutubeworks/recommendations/ 
- Airbnb基於Embedding的實時搜索推薦系統
  - 2018[Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb) 
- 阿裏巴巴深度學習推薦系統的進化
- “麻雀雖小，五臟俱全”的開源推薦系統SparrowRecSys
  - SparrowRecSys 是一個電影推薦系統
  - “麻雀雖小，五臟俱全”之意。
  - 專案是一個基於 maven 的混合語言專案，同時包含了 TensorFlow，Spark，Jetty Server 等推薦系統的不同模組。
- Meta生成式推薦模型GR的工程實現 Generative Recommendations
  - 2024`論文`[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)
  - https://github.com/meta-recsys/generative-recommenders
