#!/usr/bin/env python
# coding: utf-8

# **First Section**

# =========================================================================

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import stats
import seaborn as sns
style.use('ggplot')


# In[2]:


df = pd.read_csv('data.csv')

# In[3]:


df


# **Relational Plot (Scatter)**

# In[4]:


data = df[['age' , 'income' , 'spending_score' , 'membership_years' , 'purchase_frequency' , 'last_purchase_amount']]
data


# In[5]:


plt.rcParams.update({'font.size': 14})
x = data['age']
y = data['income']
fig, ax = plt.subplots(figsize=(9, 9))


gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
mn = np.min(x)
mx = np.max(x)
x1 = np.linspace(mn, mx, 500)
y1 = gradient * x1 + intercept
plt.plot(x, y, 'og')
plt.plot(x1, y1, '-r')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('regression plot of Income Vs. Age')
plt.show()


# **Categorical Plot (Bar Chart/Histogram/Piechart)**

# In[6]:


fig, ax = plt.subplots(figsize=(9, 9))
xf = df.groupby('preferred_category')['income'].mean().plot(kind='barh', color='orange', title='Mean Income Per Category', legend=False)
xf.bar_label(xf.containers[0], label_type='edge')


# In[7]:


plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(15, 10))
values, bins, bars=plt.hist(df['income'], color='gold', edgecolor='blue')
plt.xlabel("Income")
plt.ylabel("Number of Customers")
plt.title('Income Distrubtion')
plt.bar_label(bars, fontsize=20, color='navy')

min_ylim, max_ylim = plt.ylim()
plt.text(df['income'].mean()*0.7, max_ylim*1.0, 'Mean: {:.2f}'.format(df['income'].mean()), fontsize=14, color='red')
plt.text(df['income'].median()*1.1, max_ylim*1.0, 'Median: {:.2f}'.format(df['income'].median()), fontsize=14, color='red')

plt.margins(x=0.01, y=0.1)
plt.show()


# In[8]:


tags = df['preferred_category'].value_counts()
tags


# In[9]:



fig = plt.figure(figsize=(7,7))
index = [0, 1]
colors = ('red','green','gold','purple','olive')
wp = {'linewidth': 2, 'edgecolor': 'black'}
tags = df['preferred_category'].value_counts()
explode = (0.1, 0.1, 0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of Preferred Categories')


# In[10]:



fig = plt.figure(figsize=(7,7))
index = [0, 1]
colors = ('red','green','gold')
wp = {'linewidth': 2, 'edgecolor': 'black'}
tags = df['gender'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of gender')


# **Correlation Heatmap**

# In[11]:


round(data.corr(), 4)


# In[12]:


plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(15, 10))
sns.heatmap(round(data.corr(), 4), cmap='coolwarm', annot=True)


# **Four Main Statistical Moments**

# **Mean**

# Mean was computed as the statistical average of each of the numerical data variables.

# In[13]:


print('Mean Age =', round(np.mean(data['age']),  2))
print('Mean income =', round(np.mean(data['income']), 2))
print('Mean spending score =', round(np.mean(data['spending_score']), 2))
print('Mean membership years =', round(np.mean(data['membership_years']), 2))
print('Mean purchase frequency =', round(np.mean(data['purchase_frequency']), 2))
print('Mean last purchase amount =', round(np.mean(data['last_purchase_amount']), 2))


# **Variance**

# Variance was computed as the statistical measure of how much the data points in each numerical variable spreads out from the statistical average.

# In[14]:


print('variance of Age =', round(np.var(data['age']), 2))
print('variance of income =', round(np.var(data['income']), 2))
print('variance of spending score =', round(np.var(data['spending_score']), 2))
print('variance of membership years =', round(np.var(data['membership_years']), 2))
print('variance of purchase frequency =', round(np.var(data['purchase_frequency']), 2))
print('variance of last purchase amount =', round(np.var(data['last_purchase_amount']), 2))


# **Skewness**

# Skewness was computed as the measure of symmetry of the distortion. Negative skewness means the data is skewed to the left or negatively skewed. Positive skewness means the data is skewed to the right or positively skewed. zero skewness or skewness very close to zero means the data is normally distributed.

# In[15]:


print('Skewness of Age =', round(stats.skew(data['age']), 2))
print('Skewness of income =', round(stats.skew(data['income']), 2))
print('Skewness of spending score =', round(stats.skew(data['spending_score']), 2))
print('Skewness of membership years =', round(stats.skew(data['membership_years']), 2))
print('Skewness of purchase frequency =', round(stats.skew(data['purchase_frequency']), 2))
print('Skewness of last purchase amount =', round(stats.skew(data['last_purchase_amount']), 2))


# **Kurtosis**

# Kurtosis was computed as the measure of taoiledness of the data, either heavy-tailed or light-tailed in relation to the normal distribution. Data with high kurtosis value means the data has heavy tails, or has outliers. Data with low kurtosis means it has light tails, or does not have outliers.

# In[16]:


print('Skewness of Age =', round(stats.kurtosis(data['age']), 2))
print('Skewness of income =', round(stats.kurtosis(data['income']), 2))
print('Skewness of spending score =', round(stats.kurtosis(data['spending_score']), 2))
print('Skewness of membership years =', round(stats.kurtosis(data['membership_years']), 2))
print('Skewness of purchase frequency =', round(stats.kurtosis(data['purchase_frequency']), 2))
print('Skewness of last purchase amount =', round(stats.kurtosis(data['last_purchase_amount']), 2))

