## 資料來源
-  [Guide to All 70+ Scikit-Learn Models and When to Use Them](https://apxml.com/posts/scikit-learn-models-guide)

#### 75. Restricted Boltzmann Machines (RBM)
- Restricted Boltzmann Machines (RBMs) are generative neural network models that learn a joint distribution over the input data and hidden features. 
- They are commonly used for dimensionality reduction, feature learning, and as building blocks for deep belief networks.

- When to avoid
- Large datasets: RBMs can be computationally expensive for very large datasets.
- High-dimensional data: Training RBMs on very high-dimensional data may require significant computational resources.
- Hyperparameter tuning: RBMs are sensitive to hyperparameters like learning rate and the number of hidden units.

- Implementation
```python
from sklearn.neural_network import BernoulliRBM

rbm = BernoulliRBM(n_components=2, learning_rate=0.01, n_iter=100, random_state=42)
rbm.fit(X)
```
