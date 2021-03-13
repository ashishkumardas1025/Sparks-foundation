#!/usr/bin/env python
# coding: utf-8

# # Author : Ashish Kumar Das
# 

# # Task 1 : Prediction using Supervised Machine Learning

# # GRIP @ The Sparks Foundation
# In this regression task I tried to predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# 
# This is a simple linear regression task as it involves just two variables.  

# # Technical Stack : Sikit Learn, Numpy Array, Pandas, Matplotlib, Seaborn

# In[1]:


# Importing the required Libraries need for analysis
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# # Step 1 - Reading The Data 

# In[4]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[3]:


print(df.head())
print(df.tail())


# # Step 2 - Exploratory Data Analysis

# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.info()


# This shows concise summary of the DataFrame

# In[8]:


df.shape


# This shows number of rows and columns present in the data set

# In[9]:


print(df.isnull().sum())


# This shows the data doesn't have any Null(missing) values

# In[10]:


print(df.describe())


# This shows the descriptive statistics

# #  Step 3- Data Visualization

# In[11]:


df.plot(x= "Hours", y= "Scores" , kind = "scatter",style = "0")
plt.title("hours vs scores in percentage")
plt.xlabel("hours_studied")
plt.ylabel("score_obtained")
plt.show()


# # Step 4 - Regression Model

# In[12]:


sns.regplot(x= df["Hours"],y= df["Scores"], color= "brown")
plt.title("Regression_plot", size=25)
plt.xlabel("hours_studied")
plt.ylabel("score_obtained")
plt.show()
print(df.corr())


# This shows that data is positively correlated
# 
# now splitting the dataset into train and test data

# In[13]:


x=df[["Hours"]]
y=df["Scores"]

#splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train


# In[14]:


regression=LinearRegression()
regression.fit(x_train,y_train)


# # Step 5 - Predicting test data  

# In[15]:


y_pred=regression.predict(x_test)
y_pred


# Here y_pred is a numpy array that contains all the predicted values for the input values in the x_test series.

# In[16]:


#comparing Actual and Predicted values for x_test
df=pd.DataFrame({"Actual" : y_test, "predicted": y_pred})
df


# # Step 6 - Visualization of actual and predicted marks

# In[17]:


plt.scatter(x_test,y_test, color="brown")
plt.plot(x_test,y_pred, color="blue")
plt.xlabel("hours_studied")
plt.ylabel("Scores_of_students")
plt.title("Testing data actual values")
plt.show()

plt.scatter(y_test,y_pred, color="red")
plt.xlabel("hours_studied")
plt.ylabel("Scores_of_students")
plt.title("Testing data predicted values")
plt.show()


# # Step 7 - Model Accuracy using mean Square Error

# In[18]:


#model_accuracy
mean_abs_error= metrics.mean_absolute_error(y_test,y_pred)
print("mean_abs_error :" , mean_abs_error)
print("R-2 Score :", metrics.r2_score(y_test,y_pred))


# In[24]:


#prediction
Hours=[9.25]
solution = regression.predict([Hours])
print("Precentage of student who studied 9.25 hrs per day is: ",format(round(solution[0],2)))


# Now from above score we can comment that our model is 94% accurate
# 
# Thank You
