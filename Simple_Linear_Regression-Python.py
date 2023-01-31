#!/usr/bin/env python
# coding: utf-8

# ## Simple Linear Regression Ex 1: Swedish Car Insurance
# 
# In this exercise we will build a simple linear regression model using the number of car insurance claims in  predicting the amount paid out from the number of Swedish car insurance claims.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plot
import statsmodels.api as stats
import numpy as np


# ### Loading Our Data

# In[4]:


insurance_df = pd.read_csv('auto_insurance_sweden.csv')


# ### Assessing the Data

# In[6]:


insurance_df.shape


# In[5]:


insurance_df.head()


# ### Visualisation of the Data

# In[8]:


plot.scatter(insurance_df.claims, insurance_df.payment)
plot.xlabel('Claims')
plot.ylabel('Payment [100k Kroner]')
plot.show()


# ### Fitting the Linear Regression Model

# In[13]:


Y_insurance = insurance_df.payment

#using the stats.add.constact() function to force the model to have a value for the intercept.
X_insurance = stats.add_constant(insurance_df['claims'])


# In[14]:


#Created an instance of the model using stats.OLS
model_insurance = stats.OLS(Y_insurance, X_insurance)
results_insurance = model_insurance.fit()


# In[11]:


print(results_insurance.summary())


# ### Viewing Line Parameters

# In[12]:


#extracing specifically values of theta-0 and theta-1 from results above using params
intercept_insurance = results_insurance.params[0]
claims_coeff = results_insurance.params[1]
ssr_insurance = results_insurance.ssr

print('The intercept value is {:.3f}'.format(intercept_insurance))
print('The coefficient (slope) for the claims independent variable is {:.3f}'.format(claims_coeff))
print('The sum of square residuals is {:.1f}'.format(ssr_insurance))


# ### Plotting Results

# In[ ]:


#Prep training observations
plot.scatter(insurance_df.claims, insurance_df.payment, label='Observed')

#Prep line of best fit
plot.plot(x_synthetic, y_pred_insurance, color='k', ls='--', label='Model')
x_synthetic = np.linspace(0,insurance_df.claims.max(), 50)
y_pred_insurance = claims_coeff*x_synthetic + intercept_insurance

#plot combined chart
plot.xlabel('Claims')
plot.ylabel('Payment [100k Kroner]')
plot.legend()
plot.show()

