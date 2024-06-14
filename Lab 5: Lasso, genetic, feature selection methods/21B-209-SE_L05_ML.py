#!/usr/bin/env python
# coding: utf-8

# In[40]:


# NAME: MUHAMMAD RAYYAN KHAN
# ROLL NO: 21B-209-SE
# SECTION: SE-A


# # QUESTION 1

# In[27]:


pip install mlxtend


# In[31]:


import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

housing_data = pd.read_csv('housing.csv')

X = housing_data.drop(columns=['Price'])  # 'Price' is the target variable
y = housing_data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()

sfs = SFS(lr,
          k_features='best',
          forward=True,
          floating=False,
          verbose=2,
          scoring='r2',
          cv=5)

sfs.fit(X_train, y_train)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("Missing values in y_train:", y_train.isnull().sum())



# # QUESTION 2

# In[32]:


from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

iris = load_diabetes()
X = iris.data
y = iris.target

lr = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=42, n_jobs=-1, max_iter=500)
lr.fit(X, y)

bfs = SFS(lr,
          k_features='best',
          forward=False,
          floating=False,
          verbose=2,
          scoring='accuracy',
          cv=5) 

bfs.fit(X, y)

features = bfs.k_feature_names_
print("Selected features:", features)


# # QUESTION 3

# In[39]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import pandas as pd

iris = load_iris()
X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
y_series = iris.target

knn = KNeighborsClassifier(n_neighbors=3)

efs1 = EFS(knn, min_features=1, max_features=4, scoring='accuracy', print_progress=True, cv=5)

efs1.fit(X_df, y_series)

print(f'Best accuracy score: {efs1.best_score_:.2f}')
print(f'Best subset (indices): {efs1.best_idx_}')
print(f'Best subset (corresponding names): {efs1.best_feature_names_}')


# # QUESTION 4

# In[36]:


from sklearn.datasets import load_iris
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

iris = load_iris()
X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
y_series = iris.target

rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=2)

model = GradientBoostingClassifier()

pipe = Pipeline([('feature_selection', rfe), ('model', model)])

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)

n_scores = cross_val_score(pipe, X_df, y_series, scoring='accuracy', cv=cv, n_jobs=-1)

print("Mean Accuracy Score:", np.mean(n_scores))

pipe.fit(X_df, y_series)

selected_features = pipe.named_steps['feature_selection'].support_
print("Selected Features:", selected_features)


# # QUESTION 5

# In[37]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
])

param_grid = {'model__alpha': np.arange(0.1, 10, 0.1)}

search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=3)
search.fit(X_train, y_train)

coefficients = search.best_estimator_.named_steps['model'].coef_

importance = np.abs(coefficients)
print("Absolute coefficients:", importance)

selected_features_indices = np.where(importance > 0)[0]
print("Selected features indices:", selected_features_indices)


# # QUESTION 6

# In[38]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

feature_importances = rf.feature_importances_

print("Feature Importances:")
for i, importance in enumerate(feature_importances):
    print(f"Feature {i+1}: {importance:.4f}")


# In[ ]:




