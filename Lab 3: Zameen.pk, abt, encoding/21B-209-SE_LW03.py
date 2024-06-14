#!/usr/bin/env python
# coding: utf-8

# # LAB WORK - 03

# In[ ]:


# NAME: MUHAMMAD RAYYAN KHAN
# ROLL NUMBER: 21B-209-SE
# SECTION: SE-A


# # Question 1 and 2

# In[13]:


import pandas as pd
data = pd.read_csv('Property_with_Feature_Engineering.csv')


# In[14]:


data.dtypes


# In[15]:


data.isnull().sum()


# In[16]:


data.isna().sum()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Box plot for numerical columns to visualize outliers
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8,6))
    sns.boxplot(data[col])
    plt.title(col)
    plt.show()


# # Question 3

# In[ ]:


import pandas as pd
data = pd.read_csv('Property_with_Feature_Engineering.csv')
data.fillna(data.mean(), inplace=True)
#data.fillna(data.median(), inplace=True)
#data.fillna(data.mode().iloc[0], inplace=True)


# # Question 4

# In[5]:


import pandas as pd
data = pd.read_csv('Property_with_Feature_Engineering.csv')
selected_columns = ['property_id','location_id','page_url','property_type','price','price_bin','location','city','province_name','locality']
selected_data = data[selected_columns]
selected_data.to_csv('selected_house_data.csv', index=False)


# # Question 5

# In[6]:


# Descriptive variables
X = selected_data.drop('price', axis=1)  
# Target variable
y = selected_data['locality']  


# # Question 6

# In[7]:


import pandas as pd

selected_data = pd.read_csv('selected_house_data.csv')
numerical_stats = selected_data.describe()
categorical_stats = selected_data.describe(include=['object'])

print("Summary statistics for numerical variables:")
print(numerical_stats)

print("\nFrequency counts for categorical variables:")
print(categorical_stats)


# # Question 7

# In[8]:


import pandas as pd

selected_data = pd.read_csv('selected_house_data.csv')
covariance_matrix = selected_data.cov()
correlation_matrix = selected_data.corr()

print("Covariance Matrix:")
print(covariance_matrix)

print("\nCorrelation Matrix:")
print(correlation_matrix)


# # Question 8

# In[11]:


import pandas as pd

selected_data = pd.read_csv('Property_with_Feature_Engineering.csv')
grouped_data = selected_data.groupby(['city', 'location', 'area'])
for group, group_data in grouped_data:
    print("City:", group[0])
    print("Location:", group[1])
    print("Area:", group[2])
    print("Property ID:", group_data['property_id'].mean())
    print("\n")


# # Question 9

# In[12]:


import pandas as pd
selected_data = pd.read_csv('Property_with_Feature_Engineering.csv')

for column in selected_data.columns:
    print("Attribute:", column)
    print(selected_data[column].value_counts())
    print("\n")


# # Question 10

# In[13]:


from sklearn.preprocessing import LabelEncoder

selected_data = pd.read_csv('Property_with_Feature_Engineering.csv')
label_encoder = LabelEncoder()
selected_data['property_type_encoded'] = label_encoder.fit_transform(selected_data['property_type'])
selected_data['province_name_encoded'] = label_encoder.fit_transform(selected_data['province_name'])
print(selected_data.head())


# In[24]:


# ----------------------------------------------------------------------------------------------------------------------


# # LAB EXERCISES

# In[1]:


# Data Processing


# In[2]:


import pandas as pd
data=pd.read_csv("MotorInsuranceFraudClaimABTFull.csv")
df =pd.DataFrame(data)
df.head()


# In[14]:


import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

data = np.array([[1., 2., 3., 4.],
                 [5., 6., np.nan, 8.],
                 [10., 11., 12., np.nan]])

df = pd.DataFrame(data)

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_data = imr.fit_transform(df.values)
print(imputed_data)


# In[15]:


import pandas as pd

df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.]})

# There are two falcons and two parrots, so it's not meaningful to compute the mean of a categorical variable
# You can directly use the groupby() function without calculating the mean for 'Max Speed'
grouped_data = df.groupby(['Animal']).mean()
print(grouped_data)


# In[18]:


data2 = pd . read_csv ("diabetes.csv")
df2 = pd.DataFrame(data2)
df2 . cov ()
df2 . corr ()


# In[4]:


import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

data = np.array([[1., 2., 3., 4.],
                 [5., 6., np.nan, 8.],
                 [10., 11., 12., np.nan]])

df = pd.DataFrame(data)

# convert string and categorical values into integer values
cleanup_nums = {" property_type ": {" House ": 1 , " Flat ": 2 ," Upper Portion": 3," Lower Portion ": 4 ," Room ": 5 ," Farm House ": 6, " Penthouse ": 7} ,
" province_name ": {" Punjab ": 1 , " Sindh ": 2 , " KPK ": 3, " Balochistan ": 4 ," GB "
: 5 , " Islamabad Capital ": 6}}
encoded_df = df.replace(cleanup_nums)
encoded_df.head( n =100 )


# In[3]:


data2=pd.read_csv("diabetes.csv")
df2=pd.DataFrame(data2 ,columns=['Glucose',"BMI"])
df2.head()
df2.loc[df2['BMI'] >= 50]


# In[ ]:




