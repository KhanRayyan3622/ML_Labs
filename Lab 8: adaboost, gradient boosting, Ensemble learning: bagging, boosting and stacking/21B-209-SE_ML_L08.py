#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name: Muhammad Rayyan Khan
# Roll Number: 21B-209-SE
# Section: SE-A


# # LAB TASKS

# # Task 1

# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[3]:


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


base_regressor = DecisionTreeRegressor()


# In[6]:


bagging_regressor = BaggingRegressor(base_estimator=base_regressor, n_estimators=50, random_state=42)
bagging_regressor.fit(X_train, y_train)
bagging_pred = bagging_regressor.predict(X_test)


# In[7]:


random_forest_regressor = RandomForestRegressor(n_estimators=50, random_state=42)
random_forest_regressor.fit(X_train, y_train)
random_forest_pred = random_forest_regressor.predict(X_test)


# In[8]:


bagging_mse = mean_squared_error(y_test, bagging_pred)
random_forest_mse = mean_squared_error(y_test, random_forest_pred)


# In[9]:


print("Bagging Regressor MSE:", bagging_mse)
print("Random Forest Regressor MSE:", random_forest_mse)


# # Task 2

# In[10]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# In[11]:


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


# In[12]:


threshold = 150
y_class = (y > threshold).astype(int)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)


# In[14]:


base_classifier = DecisionTreeClassifier()

bagging_classifier = BaggingClassifier(base_estimator=base_classifier, n_estimators=50, random_state=42)
bagging_classifier.fit(X_train, y_train)
bagging_pred = bagging_classifier.predict(X_test)

random_forest_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest_classifier.fit(X_train, y_train)
random_forest_pred = random_forest_classifier.predict(X_test)


# In[15]:


bagging_accuracy = accuracy_score(y_test, bagging_pred)
random_forest_accuracy = accuracy_score(y_test, random_forest_pred)

bagging_f1 = f1_score(y_test, bagging_pred)
random_forest_f1 = f1_score(y_test, random_forest_pred)

bagging_cm = confusion_matrix(y_test, bagging_pred)
random_forest_cm = confusion_matrix(y_test, random_forest_pred)


# In[16]:


print("Bagging Classifier Accuracy:", bagging_accuracy)
print("Random Forest Classifier Accuracy:", random_forest_accuracy)

print("Bagging Classifier F1 Score:", bagging_f1)
print("Random Forest Classifier F1 Score:", random_forest_f1)

print("Bagging Classifier Confusion Matrix:")
print(bagging_cm)
print("Random Forest Classifier Confusion Matrix:")
print(random_forest_cm)


# In[ ]:





# # Task 3

# In[17]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# In[18]:


breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)
adaboost_pred = adaboost_classifier.predict(X_test)

random_forest_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest_classifier.fit(X_train, y_train)
random_forest_pred = random_forest_classifier.predict(X_test)


# In[21]:


adaboost_accuracy = accuracy_score(y_test, adaboost_pred)
random_forest_accuracy = accuracy_score(y_test, random_forest_pred)

adaboost_f1 = f1_score(y_test, adaboost_pred)
random_forest_f1 = f1_score(y_test, random_forest_pred)

adaboost_cm = confusion_matrix(y_test, adaboost_pred)
random_forest_cm = confusion_matrix(y_test, random_forest_pred)


# In[22]:


print("AdaBoost Classifier Accuracy:", adaboost_accuracy)
print("Random Forest Classifier Accuracy:", random_forest_accuracy)

print("AdaBoost Classifier F1 Score:", adaboost_f1)
print("Random Forest Classifier F1 Score:", random_forest_f1)

print("AdaBoost Classifier Confusion Matrix:")
print(adaboost_cm)
print("Random Forest Classifier Confusion Matrix:")
print(random_forest_cm)


# # Task 4

# In[23]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# In[24]:


breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)
adaboost_pred = adaboost_classifier.predict(X_test)

random_forest_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest_classifier.fit(X_train, y_train)
random_forest_pred = random_forest_classifier.predict(X_test)

adaboost_accuracy = accuracy_score(y_test, adaboost_pred)
adaboost_f1 = f1_score(y_test, adaboost_pred)
adaboost_cm = confusion_matrix(y_test, adaboost_pred)

random_forest_accuracy = accuracy_score(y_test, random_forest_pred)
random_forest_f1 = f1_score(y_test, random_forest_pred)
random_forest_cm = confusion_matrix(y_test, random_forest_pred)


# In[26]:


print("AdaBoost Classifier:")
print("Accuracy:", adaboost_accuracy)
print("F1 Score:", adaboost_f1)
print("Confusion Matrix:")
print(adaboost_cm)
print()

print("Random Forest Classifier:")
print("Accuracy:", random_forest_accuracy)
print("F1 Score:", random_forest_f1)
print("Confusion Matrix:")
print(random_forest_cm)


# # Task 5

# In[28]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier


# In[29]:


base_estimators = [
    ('adaboost', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('random_forest', RandomForestClassifier(n_estimators=50, random_state=42))
]


# In[30]:


stacking_classifier = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())
stacking_classifier.fit(X_train, y_train)
stacking_pred = stacking_classifier.predict(X_test)

stacking_accuracy = accuracy_score(y_test, stacking_pred)
stacking_f1 = f1_score(y_test, stacking_pred)
stacking_cm = confusion_matrix(y_test, stacking_pred)


# In[31]:


print("Stacking Classifier:")
print("Accuracy:", stacking_accuracy)
print("F1 Score:", stacking_f1)
print("Confusion Matrix:")
print(stacking_cm)


# # Task 6

# In[32]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier


# In[35]:


print("Stacking Classifier:")
print("Accuracy:", stacking_accuracy)
print("F1 Score:", stacking_f1)
print("Confusion Matrix:")
print(stacking_cm)


# In[36]:


adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)
adaboost_pred = adaboost_classifier.predict(X_test)

adaboost_accuracy = accuracy_score(y_test, adaboost_pred)
adaboost_f1 = f1_score(y_test, adaboost_pred)
adaboost_cm = confusion_matrix(y_test, adaboost_pred)

print("\nAdaBoost Classifier:")
print("Accuracy:", adaboost_accuracy)
print("F1 Score:", adaboost_f1)
print("Confusion Matrix:")
print(adaboost_cm)


# In[37]:


random_forest_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest_classifier.fit(X_train, y_train)
random_forest_pred = random_forest_classifier.predict(X_test)

random_forest_accuracy = accuracy_score(y_test, random_forest_pred)
random_forest_f1 = f1_score(y_test, random_forest_pred)
random_forest_cm = confusion_matrix(y_test, random_forest_pred)

print("\nRandom Forest Classifier:")
print("Accuracy:", random_forest_accuracy)
print("F1 Score:", random_forest_f1)
print("Confusion Matrix:")
print(random_forest_cm)


# # Homework - Project Assignment - 2

# In[38]:


import pandas as pd
from numpy import mean, std
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
from sklearn.ensemble import VotingClassifier


def get_dataset():
    df = pd.read_csv('diabetes.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def get_models():
    models = []
    models.append(('lr', LogisticRegression()))
    models.append(('knn', KNeighborsClassifier()))
    models.append(('tree', DecisionTreeClassifier()))
    models.append(('nb', GaussianNB()))
    models.append(('svm', SVC(probability=True)))
    return models

scoring_function = make_scorer(accuracy_score)

def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring=scoring_function, cv=cv, n_jobs=-1)
    return scores

X, y = get_dataset()
models = get_models()
results, names = [], []

for name, model in models:
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('> %s %.3f (%.3f)' % (name, mean(scores), std(scores)))

pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

def evaluate_ensemble(models, X, y):
    if len(models) == 0:
        return 0.0
    ensemble = VotingClassifier(estimators=models, voting='soft')
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(ensemble, X, y, scoring=scoring_function, cv=cv, n_jobs=-1)
    return mean(scores)

def prune_round(models_in, X, y):
    baseline = evaluate_ensemble(models_in, X, y)
    best_score, removed = baseline, None
    for m in models_in:
        dup = models_in.copy()
        dup.remove(m)
        result = evaluate_ensemble(dup, X, y)
        if result > best_score:
            best_score, removed = result, m
    return best_score, removed

def prune_ensemble(models, X, y):
    best_score = 0.0
    while True:
        score, removed = prune_round(models, X, y)
        if removed is None:
            print('no further improvement')
            break
        best_score = score
        models.remove(removed)
        print('> %.3f (removed: %s)' % (score, removed[0]))
    return best_score, models

models = get_models()
score, model_list = prune_ensemble(models, X, y)
names = ','.join([n for n, _ in model_list])
print('Models: %s' % names)
print('Final Mean Accuracy: %.3f' % score)


# # EVIDENCE

# In[ ]:


# From these results, it seems that the Support Vector Machine, Logistic Regression, and Gaussian Naive Bayes classifiers 
# perform relatively well across both datasets. However, considering the ensemble pruning, the Support Vector Machine was 
# removed, leaving Logistic Regression and Gaussian Naive Bayes.
# Given these observations, the selected model would likely be either Logistic Regression or Gaussian Naive Bayes. 
# However, since Logistic Regression slightly outperforms Gaussian Naive Bayes in both datasets and was retained in the pruned
# ensemble, it could be considered the best model overall.

