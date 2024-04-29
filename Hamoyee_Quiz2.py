#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data = pd.read_csv("C:\\Users\\jayap\\anaconda3\\BPPD\\Python for DataAnalysis\\energydata_complete.csv")


# In[6]:


data.size


# In[10]:


data.columns


# In[11]:


data.head


# In[12]:


data.shape


# In[13]:


data.dtypes.value_counts()


# In[14]:


data.describe()


# In[15]:


data.isnull().sum()


# In[16]:


data.duplicated().sum()


# In[18]:


data[["rv1", "rv2"]]


# In[19]:


assert all(data["rv1"]) == all(data["rv2"])


# In[20]:


data["Appliances"].sort_values().unique()


# In[21]:


data["Appliances"].value_counts().sort_index()


# In[22]:


data["lights"].unique()


# In[23]:


data["lights"].sort_values().value_counts()


# In[24]:


len(data["T1"].unique())


# In[25]:


data["T1"].value_counts().sort_index()


# In[26]:


len(data["RH_1"].unique())


# In[27]:


data["RH_1"].value_counts().sort_index()


# In[28]:


data["T2"].unique()


# In[29]:


data["T2"].value_counts()


# In[30]:


data["RH_2"].unique()


# In[31]:


data["RH_2"].value_counts()


# In[32]:


data["T3"].unique()


# In[33]:


data["T3"].value_counts()


# In[34]:


data["RH_3"].unique()


# In[35]:


data["RH_3"].value_counts()


# In[36]:


data["T4"].unique()


# In[37]:


data["T4"].value_counts()


# In[39]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['T2']].values
y = df['T6'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", round(rmse, 3))


# In[41]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Remove columns "date" and "lights"
data = data.drop(columns=["date", "lights"])

# Set the target variable
target_variable = "Appliances"

# Split the dataset into features (X) and target variable (y)
X = data.drop(columns=[target_variable])
y = data[target_variable]

# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values  # Convert y to numpy array for consistency

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Fit a multiple linear regression model using the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate Mean Absolute Error for the training set
mae_train = mean_absolute_error(y_train, y_train_pred)
print("Mean Absolute Error (training set):", round(mae_train, 3))



# In[43]:


y_train_pred = model.predict(X_train)

mse_train = mean_squared_error(y_train, y_train_pred)

rmse_train = np.sqrt(mse_train)
print("Root Mean Squared Error (training set):", round(rmse_train, 3))


# In[44]:


y_test_pred = model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_test_pred)
print("Mean Absolute Error (test set):", round(mae_test, 3))


# In[45]:


y_test_pred = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_pred)

rmse_test = np.sqrt(mse_test)
print("Root Mean Squared Error (test set):", round(rmse_test, 3))


# In[46]:


lasso_model = Lasso()
lasso_model.fit(X_train, y_train)

feature_weights = lasso_model.coef_

non_zero_features = sum(feature_weights != 0)
print("Number of features with non-zero feature weights:", non_zero_features)


# In[47]:


y_test_pred_lasso = lasso_model.predict(X_test)

mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)

rmse_test_lasso = np.sqrt(mse_test_lasso)
print("Root Mean Squared Error (test set) with Lasso Regression:", round(rmse_test_lasso, 3))

