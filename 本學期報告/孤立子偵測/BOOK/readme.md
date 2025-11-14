

#### [Beginning Anomaly Detection Using Python-Based Deep Learning](https://learning.oreilly.com/library/view/beginning-anomaly-detection/9798868800085/)
- https://github.com/apress/beginning-anomaly-detection-python-deep-learning-2e/tree/master
```
1. Introduction to Anomaly Detection
2. Introduction to Data Science
3. Introduction to Machine Learning
4. Traditional Machine Learning Algorithms
   - 資料集:KDDCUP 1999 dataset
   - OneClassSVM
   - IsolationForest
5. Introduction to Deep Learning
6. Autoencoders
7. Generative Adversarial Networks
8. Long Short-Term Memory Models
9. Temporal Convolutional Networks
10. Transformers
11. Practical Use Cases and Future Trends of Anomaly Detection
```
## Chapter 4 Traditional Machine Learning Algorithms
- 資料集:KDDCUP 1999 dataset
- OneClassSVM
- IsolationForest
#### OneClassSVM
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Preparation
# 
# First, we must load the data and then transform it into a usable condition to pass into the One-Class Support Vector Machine (OC-SVM) algorithm.
# 
# We define a list of columns that correspond with the data. Unlike some other data files, the data does not contain columns so we must define them ourselves to tell pandas what the column labels should be. Otherwise, they'll be numbered by numerical index, making it harder to interpret the data.

# In[2]:


columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", 
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

df = pd.read_csv("data/kddcup.data.corrected", sep=",", names=columns, index_col=None)


# This is a very large dataset. Let's only concern ourselves with http attacks

# In[3]:


# Filter to only 'http' attacks
df = df[df["service"] == "http"]
df = df.drop("service", axis=1)


# There are many types of anomalous attacks. To make things simple, we will only detect whether a data point is normal or if it is an anomaly

# In[4]:


# Label of 'normal.' becomes 0, and anything else becomes 1 and is treated as an anomaly.
df['label'] = df['label'].apply(lambda x: 0 if x=='normal.' else 1)
df['label'].value_counts()


# The next step we should perform is to encode the categorical columns. However, there are many columns to search over, so let's automate this. The following cell will create a dictionary of column names to their corresponding data type. From that data type dictionary, we can iterate and find only the columns that are strings, which carry a datatype of "object" in pandas.
# 
# Additionally, we will be applying **standard scaling** to all the numeric columns. When all the variables are on a similar scale, the one-class support vector machine can arrive at a solution much more easily. This is a crucial step for support vector machine models.

# In[5]:


datatypes = dict(zip(df.dtypes.index, df.dtypes))
encoder_map = {}
for col, datatype in datatypes.items():
    if datatype == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoder_map[col] = encoder 
    else:
        if col == 'label':
            continue 
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        encoder_map[col] = scaler 


# In[6]:


# Check the variables with highest correlation with 'label'
df2 = df.copy()
label_corr = df2.corr()['label']


# Now that we have filtered out only the columns with a stronger correlation with label, let's split our data into train-test-val splits.

# In[7]:


# Filter out anything that has null entry or is not weakly correlated
train_cols = label_corr[(~label_corr.isna()) & (np.abs(label_corr) > 0.2)]
train_cols = list(train_cols[:-1].index)
labels = df2['label']

# Conduct a train-test split    
x_train, x_test, y_train, y_test = train_test_split(df2[train_cols].values, labels.values, test_size = 0.15, random_state = 42)


# In[8]:


# Additional split of training dataset to create validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[9]:


print("Shapes")
print(f"x_train:{x_train.shape}\ny_train:{y_train.shape}")
print(f"\nx_val:{x_val.shape}\ny_val:{y_val.shape}")
print(f"\nx_test:{x_test.shape}\ny_test:{y_test.shape}")


# Since the OC-SVM only trains on data belonging to one class (since it is used for novelty / outlier detection), we need to split the dataset into one part comprised only of normal instances (to be used for training), and another part that combines a mixture of normal data and just the anomalies.

# In[10]:


# Split a set into 80% only normal data, and 20% normal data + any anomalies in set
def split_by_class(x, y):
    # Separate into normal, anomaly
    x_normal = x[y == 0]
    x_anom = x[y==1]
    
    y_normal = y[y==0]
    y_anom = y[y==1]
    
    # Split normal into 80-20 split, one for pure training and other for eval
    x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_normal, y_normal, test_size=0.2, random_state=42)
    
    # Combine the eval set with the anomalies to test outlier detection
    x_train_test = np.concatenate((x_train_test, x_anom))
    y_train_test = np.concatenate((y_train_test, y_anom))
    
    # Shuffle the eval set
    random_indices = np.random.choice(list(range(len(x_train_test))), size=len(x_train_test), replace=False)
    x_train_test = x_train_test[random_indices]
    y_train_test = y_train_test[random_indices]
    
    return x_train_train, x_train_test, y_train_train, y_train_test
                        
    


# We will split up our training dataset this way.

# In[11]:


### Train on normal data only. The _test splits have normal and anomaly data both
x_train_train, x_train_test, y_train_train, y_train_test = split_by_class(x_train, y_train)


# In[12]:


print(f"x_train_train: {x_train_train.shape}")
print(f"y_train_train: {y_train_train.shape}")   
print(f"x_train_test: {x_train_test.shape}") 
print(f"y_train_test: {y_train_test.shape}") 


# One issue with the OC-SVM is that the training time scales very poorly with data. And so, for the sake of prototyping, we will be using far fewer samples than usual to find the optimal set of hyperparameters and then scale it back up. 

# In[13]:


# nu is a cap on the upper bound of training errors and lower bound of the fraction of support vectors. We will first try to enter the expected amount of anomalies
svm = OneClassSVM(nu=0.0065, gamma=0.05)
svm.fit(x_train_train[:50000])


# Predictions returned are either:
# - 1 -> normal data (inlier)
# - -1 -> anomaly data (outlier)
# 
# So, to convert the predictions to be in terms of 0 (normal) or 1 (anomaly), we only need to apply a boolean condition of:
# 
# preds < 0
# 
# Since -1, the anomaly, is less than 0, it evaluates to True, which is converted to 1.
# Likewise, 1, the normal point, is greater than 0, which evaluates to False, which converts to 0.
# 

# In[14]:


preds = svm.predict(x_train_test)
# -1 is < 0, so it flags as 1. 1 is > 0, so flags as 0.
preds = (preds < 0).astype(int) 
preds


# In[15]:


pre = precision_score(y_train_test, preds )
rec = recall_score(y_train_test, preds)
f1 = f1_score(y_train_test, preds)

print(f"Precision: {pre}")
print(f"Recall: {rec}")
print(f"F1-Measure: {f1}")


# This looks like a really good score just evaluating on a \_test split derived from the training data. 

# In[16]:


ConfusionMatrixDisplay(confusion_matrix(y_train_test, preds)).plot()


# The confusion matrix looks good overall - there were no anomalies that were false negatives, but there were more than a few false positives. Now let's see how this works on our original test split.

preds = svm.predict(x_test)
preds = (preds < 0).astype(int) # -1 is < 0, so it flags as 1. 1 is > 0, so flags as 0.
preds


pre = precision_score(y_test, preds )
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Precision: {pre}")
print(f"Recall: {rec}")
print(f"F1-Measure: {f1}")


# It seems like our model massively overfit to the training data, since the F1-measure is so low compared to earlier. It makes sense that the performance looked good on the x_train_test split because this was derived from the training data, and thus has a similar distribution. The test split was completely different and was never involved in the training process.
# 
# We now need to look at hyperparameter tuning to see if we can get this model to generalize better.
# 
# Of interest are the following hyperparameter settings:
# - **gamma**: A parameter used in the RBF kernel function. It determines the radius of influence that points have on each other, with smaller gamma translating to larger radii. We want the radius of influence to be as large as possible to include all the outliers, but not too large, or else the model ends up learning nothing important. If gamma is too large, then the SVM will try and fit the training data more closely, which might lead to overfitting. This makes gamma an important parameter to tune.
# 
# - **nu**: This is an upper bound on the fraction of training errors allowed and a lower bound on the fraction of support vectors. It is a more sensitive hyperparameter than gamma and it can influence the output behavior of the model quite a bit, so it is very important to tune this value according to your dataset. You may also need to retune this even if your training data size changes. A very small nu places a stricter requirement on how many training errors are allowed, while a larger nu value allows for more training errors to slip through when training the model.
# 


# Given an svm instance, x, and y data, train and evaluate and return results
def experiment(svm, x_train, y_train, x_test, y_test):
    
    # Fit on the training data, predict on the test
    svm.fit(x_train)
    
    preds = svm.predict(x_test)
    
    # Predictions are either -1 or 1
    preds = (preds < 0).astype(int)
    
    pre = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    return {'precision': pre, 'recall': rec, 'f1': f1}
    
# Perform experimental search for best gamma parameter
validation_results = {}
gamma = [0.005, 0.05, 0.5]
for g in gamma:
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=0.0065, gamma=g)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[g] = res
    
# Printing out the results of the validation. 
[(f, validation_results[f]['f1']) for f in validation_results.keys()]

# Perform experimental search for gamma with a narrower range. Looks like smaller gamma is better
validation_results = {}
# Search 1e-5, 5e-5, 1e-4, 1.5e-4, 1e-3, 1.5e-3 and 2e-3
gamma = [1, 5, 10, 15, 20, 100, 150, 200]
for g in gamma:
    g = g / 100000.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=0.0065, gamma=g)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[g] = res

# Printing out the results of the validation.
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# Let's try the gamma value of 1e-5 and now tune **nu**.

# Perform experimental search for nu
validation_results = {}
nu = range(1, 10)
for n in nu:
    n = n / 1000.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=n, gamma=0.00001)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[n] = res
    
# Printing out the results of the validation. 
[(f, validation_results[f]['f1']) for f in validation_results.keys()]

# Perform experimental search for nu with a finer range
validation_results = {}
nu = range(1, 11)#[0.005, 0.0065, 0.01, 0.015]
for n in nu:
    n = n / 10000.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    svm = OneClassSVM(nu=n, gamma=0.00001)
    
    res = experiment(svm, x_train_train[:20000], y_train_train[:20000], x_val, y_val)
    validation_results[n] = res
                           
# Printing out the results of the validation. The optimal setting is between 512 and 2048
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# Now that we have found the optimal nu setting of 0.0002, let's try and train the model.
# 
# It turns out that gamma = 1e-5 was not performing that well once the data size increased. We upped it to 0.005, which produced favorable results and allowed the model to generalize better.
# We increased gamma back to 0.005 as it helped with the fit once we increased the number of samples.

svm = OneClassSVM(nu=0.0002, gamma=0.005)
svm.fit(x_train_train[:])


# In[29]:


preds = svm.predict(x_test)
preds = (preds < 0).astype(int)


# In[30]:


pre = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Precision: {pre}")
print(f"Recall: {rec}")
print(f"F1-Measure: {f1}")


# In[31]:

ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot()

```

#### IsolationForest

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Preparation
# 
# First, we must load the data and then transform it into a usable condition to pass into the isolation forest algorithm.
# 
# We define a list of columns that correspond with the data. Unlike some other data files, the data does not contain columns so we must define them ourselves to tell pandas what the column labels should be. Otherwise, they'll be numbered by numerical index, making it harder to interpret the data.

# In[2]:


columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", 
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

df = pd.read_csv("data/kddcup.data.corrected", sep=",", names=columns, index_col=None)


# This is a very large dataset. Let's only concern ourselves with http attacks

# In[3]:


df.shape


# In[4]:


# Filter to only 'http' attacks
df = df[df["service"] == "http"]
df = df.drop("service", axis=1)


# In[5]:


df.shape


# There are many types of anomalous attacks. To make things simple, we will only detect whether a data point is normal or if it is an anomaly

# In[6]:


df["label"].value_counts()


# In[7]:


# Label of 'normal.' becomes 0, and anything else becomes 1 and is treated as an anomaly.
df['label'] = df['label'].apply(lambda x: 0 if x=='normal.' else 1)
df['label'].value_counts()


# The next step we should perform is to encode the categorical columns. However, there are many columns to search over, so let's automate this. The following cell will create a dictionary of column names to their corresponding data type. From that data type dictionary, we can iterate and find only the columns that are strings, which carry a datatype of "object" in pandas.

# In[8]:


datatypes = dict(zip(df.dtypes.index, df.dtypes))


# In[9]:


encoder_map = {}
for col, datatype in datatypes.items():
    if datatype == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoder_map[col] = encoder 


# Let's now check the correlation matrix with respect to the label column and filter out all the columns that are not at least weakly correlated. Remember that linear correlation can be either positive or negative, so we must filter the columns by the magnitude of their correlation.

# In[10]:


# Check the variables with highest correlation with 'label'
df2 = df.copy()
label_corr = df2.corr()['label']


# In[11]:


# Filter out anything that has null entry or is not weakly correlated
train_cols = label_corr[(~label_corr.isna()) & (np.abs(label_corr) > 0.2)]
train_cols = list(train_cols[:-1].index)
train_cols


# In[12]:


# Checking the correlations
label_corr[train_cols]


# Now that we have filtered out only the columns with a stronger correlation with label, let's split our data into train-test-val splits.

# In[13]:


labels = df2['label']
# Conduct a train-test split    
x_train, x_test, y_train, y_test = train_test_split(df2[train_cols].values, labels.values, test_size = 0.15, random_state = 42)


# In[14]:


# Additional split of training dataset to create validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[15]:


print("Shapes")
print(f"x_train:{x_train.shape}\ny_train:{y_train.shape}")
print(f"\nx_val:{x_val.shape}\ny_val:{y_val.shape}")
print(f"\nx_test:{x_test.shape}\ny_test:{y_test.shape}")


# In[16]:


# Let's try out isolation forest with stock parameters. It may take a bit of time to train.
isolation_forest = IsolationForest(random_state=42)


# In[17]:


isolation_forest.fit(x_train)
anomaly_scores = isolation_forest.decision_function(x_train)


# In[18]:


plt.figure(figsize=(10, 5))
plt.hist(anomaly_scores, bins=100)
plt.xlabel('Average Path Lengths')
plt.ylabel('Number of Data Points')
plt.show()


# The anomaly scores are a function of the average path lengths in the forest assigned to each data point. The more negative the path length, the easier the sample was to isolate. Thus, if we look at our histogram, we can clearly see some isolated clump of outliers. Though it may be tempting to assign the threshold to be -0.3 and be done with it, that won't net you proper results.
# 
# Run the following cell and try other thresholds, you will find that the F1-Measure (the proper metric to optimize as it would ensure we adequately capture the population of anomalies and that we make accurate predictions) does not seem to be able to reach a proper score.
# 
# A score that is at least 0.9 is desirable for an F1-Measure. 

# In[19]:


threshold = -0.3
anomalies = anomaly_scores < threshold

precision = precision_score(y_train, anomalies)
recall = recall_score(y_train, anomalies)
f1 = f1_score(y_train, anomalies)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Measure: {f1}")


# So what now, is that it for the isolation forest? No, we haven't even begun the hyperparameter tuning validation process. 
# 
# The isolation forest depends heavily on its hyperparameter settings to guide how it decides to build the trees and partition the data. You must tune it for the task. With that being said, let's look at the parameters we want to tune:
# 
# - **max_samples**: how many samples to use to train each tree in the forest
# - **max_features**: how many features per data point to use to train each tree in the forest
# - **n_estimators**: how many trees should comprise the forest
# 
# There is also a feature, **contamination**, which tells the isolation forest what proportion of the data is contaminated by anomalies. But for this, we can calculate the proportion from our training set and pass it in directly.

# ### Hyperparameter Tuning with Hold-out Validation
# 
# Let's start the validation process. First, we want to tune max_samples. We do not know what range to look for, so let's go by powers of two starting from 512 and ending at 2048.
# 
# The following script is the validation framework we will be using. We will be using hold-out validation and employing our validation split that we created earlier. To be thoroughly experimentally sound, k-fold cross validation is better to use, but we just want to get a quick idea of what kind of parameters to be setting. Additionally, the validation set is a bit large  with  around ~100k records, so it should be sufficiently large to operate on directly. K-Fold is more useful when you don't have as much data to work with. 
# 
# Instead of manually setting the threshold, we are using **logistic regression** to help us predict whether a datapoint is an anomaly or not. This is a more automated method than manually experimenting with the threshold and trying out different values. However, this does insert another link in the chain that may need to be hyperparameter tuned, but we will leave it stock and as an exercise to you to see how much better you can get the performance overall.

# In[20]:


# estimate proportion of anomalies. It's about 0.0065, 0.0066
(y_train==1).sum() / len(y_train)


# In[21]:


# Given an isolation_forest instance, x, and y data, train and evaluate and return results
def experiment(isolation_forest, x, y):
    isolation_forest.fit(x)
    
    anomaly_scores = isolation_forest.decision_function(x)
    
    # Using a stock Logistic Regression model to predict labels
    lr = LogisticRegression()
    lr.fit(anomaly_scores.reshape(-1, 1), y)
    
    preds = lr.predict(anomaly_scores.reshape(-1,1))
    
    pre = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    
    return {'precision': pre, 'recall': rec, 'f1': f1}
    


# #### Tuning max_samples

# In[22]:


# Perform experimental search for max_samples
validation_results = {}
max_samples = [2**f for f in [8, 9, 10, 11, 12]]
for max_sample in max_samples:
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    isolation_forest = IsolationForest(n_estimators=50, 
                                       max_samples = max_sample, 
                                       n_jobs=-1, 
                                       contamination = 0.0066, 
                                       random_state=42)
    
    res = experiment(isolation_forest, x_val, y_val)
    validation_results[max_sample] = res
    
                       


# In[23]:


# Printing out the results of the validation. The optimal setting is between 512 and 4096
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# The results already look excellent. An f1-measure of 0.95 is already well beyond what we were seeing earlier. Let's see if we can specify a finer range and discover a better hyperparameter setting.

# In[24]:


# Repeat validation with a narrower range
validation_results = {}
max_samples = range(500, 2200, 200)
for max_sample in max_samples:
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    isolation_forest = IsolationForest(n_estimators=50, 
                                       max_samples = max_sample, 
                                       n_jobs=-1, 
                                       contamination = 0.0066, 
                                       random_state=42)
    
    res = experiment(isolation_forest, x_val, y_val)
    validation_results[max_sample] = res


# In[25]:


# Printing out the results of the validation. The optimal setting is between 512 and 2048
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# The optimal ranges are between 900 and 1100, and somewhere around 1300-1700 with 1500 being the peak. Let's explore both these ranges. We will go from 950 to 1100 in increments of 10, and 1300-1700 in increments of 100.

# In[26]:


# Repeat validation with a narrower range
validation_results = {}
max_samples = list(range(950, 1110, 10)) + list(range(1300, 1800, 100))
for max_sample in max_samples:
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    isolation_forest = IsolationForest(n_estimators=50, 
                                       max_samples = max_sample, 
                                       n_jobs=-1, 
                                       contamination = 0.0066, 
                                       random_state=42)
    
    res = experiment(isolation_forest, x_val, y_val)
    validation_results[max_sample] = res


# In[27]:


# Printing out the results of the validation. The optimal setting is between 512 and 2048
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# We can keep going with the hyperparameter tuning or be satisfied with the values we are seeing. After some point, you will see diminishing gains, so keep that in mind in the future. From our results, it seems that the setting of 950 actually performed the best somehow, so we will use this setting as we go forward with tuning the other hyperparameters.

# #### Tuning max_features

# In[28]:


# Tuning max_features
validation_results = {}
max_features = range(5, 8, 1)
for max_feature in max_features:
    max_feature = max_feature / 10.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    isolation_forest = IsolationForest(n_estimators=50, 
                                       max_samples = 950,
                                       max_features = max_feature,
                                       n_jobs=-1, 
                                       contamination = 0.0066, 
                                       random_state=42)
    
    res = experiment(isolation_forest, x_val, y_val)
    validation_results[max_feature] = res


# In[29]:


# Printing out the results of the validation. The optimal setting is between 512 and 2048
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# Our performance is increasing more. Let's narrow the range to be between 0.55 and 0.7

# In[30]:


# Tuning max_features with a narrower range
validation_results = {}
max_features = range(55, 71, 1)
for max_feature in max_features:
    max_feature = max_feature / 100.0
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    isolation_forest = IsolationForest(n_estimators=50, 
                                       max_samples = 950,
                                       max_features = max_feature,
                                       n_jobs=-1, 
                                       contamination = 0.0066, 
                                       random_state=42)
    
    res = experiment(isolation_forest, x_val, y_val)
    validation_results[max_feature] = res


# In[31]:


# Printing out the results of the validation. The optimal setting is between 512 and 2048
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# Let's stick with 0.55 and tune our final hyperparameter, the number of estimators in the forest.

# In[32]:


# Tuning max_features with a narrower range
validation_results = {}
n_estimators = range(10, 110, 10)
for n_estimator in n_estimators:
    
    # We are fixing the n_estimators to 50 to be quicker. n_jobs = -1 lets us train on all cores
    isolation_forest = IsolationForest(n_estimators=n_estimator, 
                                       max_samples = 950,
                                       max_features = 0.55,
                                       n_jobs=-1, 
                                       contamination = 0.0066, 
                                       random_state=42)
    
    res = experiment(isolation_forest, x_val, y_val)
    validation_results[n_estimator] = res


# In[33]:


# Printing out the results of the validation. The optimal setting is between 512 and 2048
[(f, validation_results[f]['f1']) for f in validation_results.keys()]


# It appears that with as little as 10 estimators in the forest, the isolation forest is able to predict anomalies quite well on the validation set. Let's keep all these hyperparameter settings, train on the training data, and evaluate on the test data and see what we get.

# In[34]:


# Set our hyperparameters that we discovered during validation and train
isolation_forest = IsolationForest(n_estimators=10, 
                                   max_samples = 950,
                                   max_features = 0.55,
                                   n_jobs=-1, 
                                   contamination = 0.0066, 
                                   random_state=42)

isolation_forest.fit(x_train)
anomaly_scores = isolation_forest.decision_function(x_train)


lr = LogisticRegression()
lr.fit(anomaly_scores.reshape(-1, 1), y_train)

preds = lr.predict(anomaly_scores.reshape(-1,1))

precision = precision_score(y_train, preds)
recall = recall_score(y_train, preds)
f1 = f1_score(y_train, preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Measure: {f1}")


# Those are strong results, especially when compared to our stock training run. Let's see how it does on the test set now.

# In[35]:


anomaly_scores = isolation_forest.decision_function(x_test)

preds = lr.predict(anomaly_scores.reshape(-1,1))

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Measure: {f1}")


# An F1-Measure of 0.979 is quite strong. Let's check the ROC-AUC score as well

# In[39]:


roc_auc_score(y_test, preds)


# This is a very strong score. Though the data is imbalanced (and thus, f1-measure is better to prioritize), this does indicate strength in overall performance over normal and anomaly predictions.
# 
# Let's examine the confusion matrix using scikit-learn's visualizer:

# In[36]:


conf_mat = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(conf_mat).plot()
plt.show()


# In[37]:


tn, fp, fn, tp = conf_mat.ravel()
print("True Negative:", tn)
print("False Negative:", fn)
print("True Positive:", tp)
print("False Positive:", fp)


```
