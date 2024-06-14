#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Name: Muhammad Rayyan Khan
# Roll Number: 21B-209-SE
# Section: SE-A


# # Question 1

# In[6]:


from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot as plt
import numpy as np


# In[7]:


def get_dataset():
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
    return X, y


# In[8]:


def get_models():
    models = []
    models.append(('lr', LogisticRegression()))
    models.append(('knn', KNeighborsClassifier()))
    models.append(('tree', DecisionTreeClassifier()))
    models.append(('nb', GaussianNB()))
    models.append(('svm', SVC(probability=True)))
    return models


# In[9]:


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# In[10]:


X, y = get_dataset()
models = get_models()


# In[11]:


results, names = [], []
for name, model in models:
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print(f">{name}: {np.mean(scores):.3f} ({np.std(scores):.3f})")


# In[12]:


plt.boxplot(results, labels=names, showmeans=True)
plt.show()


# In[14]:


ensemble = VotingClassifier(estimators=models, voting='soft')


# In[15]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[16]:


scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[17]:


print(f"Mean Accuracy: {np.mean(scores):.3f} ({np.std(scores):.3f})")


# # Question 2

# In[41]:


from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import numpy as np


# In[42]:


def load_diabetes_data():
    data = load_diabetes()
    X, y = data.data, data.target
    return X, y


# In[43]:


def evaluate_models(X, y):
    models = [
        ('Logistic Regression', LogisticRegression()),
        ('KNN', KNeighborsClassifier()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('SVM', SVC(probability=True))
    ]
    results = []
    for name, model in models:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        results.append((name, np.mean(scores), np.std(scores)))
        print(f"{name}: Mean Accuracy: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
    return results


# In[44]:


def create_voting_ensemble():
    models = [
        ('lr', LogisticRegression()),
        ('knn', KNeighborsClassifier()),
        ('tree', DecisionTreeClassifier()),
        ('nb', GaussianNB()),
        ('svm', SVC(probability=True))
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble


# In[45]:


def evaluate_ensemble(X, y, ensemble):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"Ensemble: Mean Accuracy: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
    return np.mean(scores), np.std(scores)


# In[46]:


def select_best_model(results, ensemble_score):
    best_model = max(results, key=lambda x: x[1])
    if ensemble_score[0] > best_model[1]:
        print(f"Best model: Ensemble (Mean Accuracy: {ensemble_score[0]:.3f}, Std: {ensemble_score[1]:.3f})")
    else:
        print(f"Best model: {best_model[0]} (Mean Accuracy: {best_model[1]:.3f}, Std: {best_model[2]:.3f})")


# In[47]:


def main():
    X, y = load_diabetes_data()
    results = evaluate_models(X, y)
    
    ensemble = create_voting_ensemble() 
    ensemble_score = evaluate_ensemble(X, y, ensemble)
    
    select_best_model(results, ensemble_score)

if __name__ == "__main__":
    main()


# # Homework - Project Assignment - 1

# In[55]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[57]:


df = pd.read_excel('seer_dataset.xlsx')


# In[60]:


print(df.columns)


# In[61]:


import pandas as pd

data = pd.read_excel("seer_dataset.xlsx", skiprows=1, header=None)
column_names = ['Name', 'Research', 'Research Limited-Field', 'Research Plus Limited-Field', 
                'Available in Case Listing', 'NAACCR Item #', 'Description', 'Category name']
data.columns = column_names

print(data.head())

print(data['Category name'].unique())


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[63]:


models = [
    ('lr', LogisticRegression()),
    ('knn', KNeighborsClassifier()),
    ('tree', DecisionTreeClassifier()),
    ('nb', GaussianNB()),
    ('svm', SVC(probability=True))
]


# In[64]:


for name, model in models:
    model.fit(X_train, y_train)


# In[65]:


for name, model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.3f}')


# In[66]:


ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train, y_train)


# In[67]:


y_pred_ensemble = ensemble.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Method Accuracy: {accuracy_ensemble:.3f}')

