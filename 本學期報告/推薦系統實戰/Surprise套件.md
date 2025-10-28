## Surprise套件
- [Surprise · A Python scikit for recommender systems](https://surpriselib.com/)
  - Surprise | Simple Python RecommendatIon System Engine
  - https://surprise.readthedocs.io/en/stable/
  - https://github.com/AmolMavuduru/SurprisePythonExamples
  - https://blog.csdn.net/qq_24831889/article/details/102650264
  - https://blog.csdn.net/qq_41185868/article/details/134971067
- Surprise 範例
  - https://zhuanlan.zhihu.com/p/352181306
  - pip install scikit-surprise

## 範例
```python
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```
