#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name: Muhammad Rayyan Khan
# Roll Number: 21B-209-SE
# Section: SE-A


# ## TASK-01

# In[7]:


import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, dataset, target_column):
        self.dataset = dataset
        self.target_column = target_column
        self.target_probabilities = {}
        self.conditional_probabilities = []

    def calculate_target_probabilities(self):
        total_rows = self.dataset.shape[0]
        target_classes = self.dataset[self.target_column].unique()

        for target_class in target_classes:
            target_class_count = self.dataset[self.dataset[self.target_column] == target_class].shape[0]
            self.target_probabilities[target_class] = target_class_count / total_rows

        return self.target_probabilities

    def calculate_conditional_probabilities(self, feature_column):
        total_rows = self.dataset.shape[0]
        target_classes = self.dataset[self.target_column].unique()
        feature_classes = self.dataset[feature_column].unique()
        conditional_probabilities = []

        for target_class in target_classes:
            target_class_count = self.dataset[self.dataset[self.target_column] == target_class].shape[0]
            for feature_class in feature_classes:
                count = 0
                for i in range(total_rows):
                    if (self.dataset.loc[i, feature_column] == feature_class) and (self.dataset.loc[i, self.target_column] == target_class):
                        count += 1
                conditional_probabilities.append((feature_column, feature_class, target_class, round(count / target_class_count, 3)))

        return conditional_probabilities

    def calculate_all_conditional_probabilities(self):
        for feature_column in self.dataset.columns:
            if feature_column != self.target_column:
                conditional_prob_list = self.calculate_conditional_probabilities(feature_column)
                self.conditional_probabilities += conditional_prob_list
        return self.conditional_probabilities

    def fit(self):
        self.calculate_target_probabilities()
        self.calculate_all_conditional_probabilities()
        return None

    def predict_probabilities(self, query):
        query_length = len(query)
        expected_args = len(self.dataset.columns) - 1  # excluding target column

        if query_length != expected_args:
            print('Query requires', expected_args, 'arguments')
            print('Dataset columns:', self.dataset.columns)
            return

        for i, value in enumerate(query):
            if value not in self.dataset.iloc[:, i].unique():
                print(value, 'is not in column', self.dataset.columns[i])
                return

        probability_dict = {}
        probability_product = 1

        for target_class in self.dataset[self.target_column].unique():
            for i in range(query_length):
                for item in self.conditional_probabilities:
                    if item[0] == self.dataset.columns[i] and item[1] == query[i] and item[2] == target_class:
                        probability_product *= item[3]
            probability_dict[target_class] = round(probability_product * self.target_probabilities[target_class], 4)
            probability_product = 1

        return probability_dict

    def predict(self, query):
        probabilities = self.predict_probabilities(query)
        if probabilities:
            return max(probabilities, key=probabilities.get)
        return None

    def display_probabilities(self):
        print('Target probabilities:', self.target_probabilities)
        print('Conditional probabilities:', self.conditional_probabilities)
        return None

    def calculate_conditional_probabilities_with_smoothing(self, feature_column, smoothing_factor):
        total_rows = self.dataset.shape[0]
        target_classes = self.dataset[self.target_column].unique()
        feature_classes = self.dataset[feature_column].unique()
        conditional_probabilities = []

        for target_class in target_classes:
            target_class_count = self.dataset[self.dataset[self.target_column] == target_class].shape[0]
            feature_class_count = len(feature_classes)
            for feature_class in feature_classes:
                count = 0
                for i in range(total_rows):
                    if (self.dataset.loc[i, feature_column] == feature_class) and (self.dataset.loc[i, self.target_column] == target_class):
                        count += 1
                smoothed_probability = (count + smoothing_factor) / (target_class_count + (smoothing_factor * feature_class_count))
                conditional_probabilities.append((feature_column, feature_class, target_class, round(smoothed_probability, 3)))

        return conditional_probabilities

data = {
    'CreditHistory': ['current', 'paid', 'arrears', 'none', 'current', 'paid', 'arrears', 'none'],
    'CoApplicant': ['none', 'guarantor', 'coapplicant', 'none', 'none', 'guarantor', 'coapplicant', 'none'],
    'Accommodation': ['own', 'rent', 'free', 'own', 'rent', 'free', 'own', 'rent'],
    'Target': [True, True, True, True, False, False, False, False]
}

df = pd.DataFrame(data)

nb_classifier = NaiveBayesClassifier(df, 'Target')

nb_classifier.fit()

nb_classifier.display_probabilities()

query = ['current', 'none', 'own']
print('Prediction for query', query, ':', nb_classifier.predict(query))


# ## TASK-02

# In[8]:


import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    def __init__(self, dataset, target_column):
        self.dataset = dataset
        self.target_column = target_column
        self.statistics = {}
        self.priors = {}

    def fit_model(self):
        target_classes = self.dataset[self.target_column].unique()
        for target_class in target_classes:
            subset = self.dataset[self.dataset[self.target_column] == target_class]
            self.statistics[target_class] = {
                'mean': subset.drop(columns=[self.target_column]).mean(),
                'std': subset.drop(columns=[self.target_column]).std()
            }
            self.priors[target_class] = len(subset) / len(self.dataset)

    def calculate_gaussian_probability(self, x, mean, std):
        exponent = np.exp(- ((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def calculate_class_probabilities(self, query):
        probabilities = {}
        for target_class, stats in self.statistics.items():
            probabilities[target_class] = self.priors[target_class]
            for feature in self.dataset.columns.drop(self.target_column):
                mean = stats['mean'][feature]
                std = stats['std'][feature]
                probabilities[target_class] *= self.calculate_gaussian_probability(query[feature], mean, std)
        return probabilities

    def predict(self, query):
        probabilities = self.calculate_class_probabilities(query)
        return max(probabilities, key=probabilities.get)

    def predict_probabilities(self, query):
        return self.calculate_class_probabilities(query)

data = {
    'Feature1': [3.5, 3.5, 3.5, 7.5, 4.1, 5.4, 4.2, 6.0],
    'Feature2': [2.6, 1.5, 4.9, 7.1, 6.9, 6.4, 8.2, 4.8],
    'Target': [0, 1, 0, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

nb_classifier = NaiveBayesClassifier(df, 'Target')
nb_classifier.fit_model()

query = {'Feature1': 3.0, 'Feature2': 9.0}
print('Prediction for query', query, ':', nb_classifier.predict(query))
print('Class probabilities for query', query, ':', nb_classifier.predict_probabilities(query))

