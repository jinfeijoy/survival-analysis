#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
import sys
import os
import numpy as np
sns.set_palette("hls", 8)

data_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\data_2021'



# In[2]:


data = pd.read_csv(os.path.join(data_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
data['Churn'] = data['Churn'].apply(lambda x: np.where(x=='Yes',1,0))
data['entry_age'] = 0
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
data = data[data.tenure>data.entry_age]
data.head(3)


# In[23]:


plt.figure(figsize=(8,8))
sns.barplot(data=data, x='Churn', y='tenure')


# ### K-M Plot

# In[25]:


kmf = KaplanMeierFitter()

kmf.fit(data.tenure, 
        data.Churn, 
        entry = data.entry_age,
        label = 'Kaplan Meier Estimate, full sample')

kmf.plot(linewidth=4, figsize=(12, 6))
plt.title('Customer Churn: Kaplan-Meier Curve')
plt.xlabel('Months')
plt.ylabel('Survival probability')


# ### K-M Plot for various subsamples

# In[26]:


data.columns


# In[27]:


# Let's look at the 'multiple services variable'
df1 = data[data.MultipleLines=='Yes']
df2 = data[data.MultipleLines=='No']
fig, ax = plt.subplots(1,2, figsize=(16,8))
df1.groupby('Churn')['tenure'].plot(kind='hist', ax=ax[0], title='Customers with Multiple Services.')
ax[0].legend(labels=['Not churned', 'Churned'])
df2.groupby('Churn')['tenure'].plot(kind='hist', ax=ax[1], title='Customers with Single Service.')


# In[34]:


T = data['tenure']
E = data['Churn']


# In[41]:


seniorCitizen = (data['SeniorCitizen'] == 1)

kmf.fit(T[~seniorCitizen], E[~seniorCitizen], label = 'Not Senior Citizens')
ax = kmf.plot(figsize=(12, 6))
# kmf.plot_cumulative_density()

kmf.fit(T[seniorCitizen], E[seniorCitizen], label = 'Senior Citizens')
ax = kmf.plot(ax=ax)
# kmf.plot_cumulative_density()
plt.title('Number of Services and Churn: Kaplan-Meier Curve')
plt.xlabel('Months')
plt.ylabel('Survival probability')


# ### Log-Rank Test

# In[16]:


logrank = logrank_test(data[data.SeniorCitizen==0]['tenure'], 
             data[data.SeniorCitizen==1]['tenure'], 
             event_observed_A=data[data.SeniorCitizen==0]['Churn'], 
             event_observed_B=data[data.SeniorCitizen==1]['Churn'])
logrank.print_summary()
print(logrank.p_value)        
print(logrank.test_statistic)


# ### Cox-PH Model

# In[86]:


cols_of_interest = ['SeniorCitizen', 
                    'Partner', 
                    'Dependents', 
                    'tenure', 
#                     'PhoneService', 'MultipleLines', 'InternetService', 
#                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
#                     'TechSupport', 'StreamingTV', 'StreamingMovies', 
#                     'Contract', 'PaperlessBilling', 'PaymentMethod', 
                    'MonthlyCharges', 
                    'TotalCharges', 
                    'gender', 'Churn', 'entry_age']
df = data[cols_of_interest]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.head()


# In[87]:


df = pd.get_dummies(df)
# df.drop('Contract_Two year', axis = 1, inplace = True)
# df.drop('PaymentMethod_Mailed check', axis = 1, inplace = True)
df = df.drop(['Dependents_Yes','Partner_Yes','gender_Male'], axis = 1)
df.head()


# In[109]:


cph = CoxPHFitter()
cph.fit(df, 'tenure', event_col = 'Churn', entry_col = 'entry_age', show_progress = False)
cph.print_summary()
# cph.print_summary(style='ascii')


# In[113]:


cph = CoxPHFitter()
cph.fit(df.drop(['SeniorCitizen','TotalCharges'], axis = 1), 'tenure', event_col = 'Churn', entry_col = 'entry_age', show_progress = False)
cph.print_summary()
# cph.print_summary(style='ascii')


# In[90]:


cph.plot() 


# In[114]:


cph.params_


# In[96]:


cph.plot_partial_effects_on_outcome(['SeniorCitizen','Dependents_No'], [
                                [0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1],
                            ], plot_baseline=False, figsize=(10, 6), lw=4) 


# In[106]:


prediction = cph.predict_survival_function(df)
import matplotlib.pyplot as plt
plt.plot(prediction[0].values)


# In[107]:


cph.predict_median(df)


# In[108]:


cph.predict_partial_hazard(df)


# ### Survival Tree
# 
# Reference: https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html
# 
# ### C-Index
# 
# Reference: https://medium.com/analytics-vidhya/concordance-index-72298c11eac7

# ### Aalen Additive Fitter
# 
# https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#aalen-s-additive-model

# In[125]:


data.head(3)


# In[156]:


from lifelines import AalenAdditiveFitter
aaf = AalenAdditiveFitter(coef_penalizer=0.2, fit_intercept=False)
aaf.fit(data, 'tenure', event_col='Churn', formula='SeniorCitizen + Partner + Dependents + gender + MonthlyCharges + TotalCharges + TechSupport + StreamingTV + PaperlessBilling')


# In[157]:


aaf.cumulative_hazards_.head()


# In[160]:


aaf.print_summary()


# In[162]:


aaf.plot(columns=['Dependents[T.Yes]', 'Intercept', 'SeniorCitizen','MonthlyCharges'], iloc=slice(1,15))


# In[174]:


print('median\n',aaf.predict_median(data).head(3))
print('percentile10\n',aaf.predict_percentile(data,0.1).head(3))
print('expectation\n',aaf.predict_expectation(data).head(3))
print('cumulative_hazard\n',aaf.predict_cumulative_hazard(data).head(3))
print('survival function\n',aaf.predict_survival_function(data).head(3))


# ### Weibull AFT
# 
# More details can be found from:
# * https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html
# * https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#the-weibull-aft-model

# In[3]:


cols_of_interest = ['SeniorCitizen', 
                    'Partner', 
                    'Dependents', 
                    'tenure', 
                    'MonthlyCharges', 
                    'TotalCharges', 
                    'gender', 'Churn']
df = data[cols_of_interest]
df = pd.get_dummies(df)
df = df.drop(['Dependents_Yes','Partner_Yes','gender_Male'], axis = 1)
df.head()
df.head()


# In[4]:


from lifelines import WeibullAFTFitter

aft = WeibullAFTFitter()
aft.fit(df, duration_col='tenure', event_col='Churn')

aft.print_summary(3)  # access the results using aft.summary


# In[11]:


aft.plot()


# In[12]:


df.TotalCharges.describe()


# In[18]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

# times = np.arange(0, 100)
# wft_model_rho = WeibullAFTFitter().fit(rossi, 'week', 'arrest', ancillary=True, timeline=times)
aft.plot_partial_effects_on_outcome('TotalCharges', range(0, 9000, 1000), cmap='coolwarm', ax=ax)
ax.set_title("Total Charges")


# In[5]:


X = df.loc[:10]

cumulative_hazard = aft.predict_cumulative_hazard(X, ancillary=X)
survival_fun = aft.predict_survival_function(X, ancillary=X)
median = aft.predict_median(X, ancillary=X)
percentile_90 = aft.predict_percentile(X, p=0.9, ancillary=X)
expectation = aft.predict_expectation(X, ancillary=X)

print('cumulative_hazard\n',cumulative_hazard.head(3))
print('survival_fun\n',survival_fun.head(3))
print('median\n',median.head(3))
print('percentile_90\n',percentile_90.head(3))
print('expectation\n',expectation.head(3))


# In[60]:


from numpy import exp
def calculate_survival_fun(data, t, model):
    parameters = pd.DataFrame(model.summary.coef).reset_index()
    rho = exp(parameters[parameters.param == 'rho_'].coef.values[0])
    almbda_inter = parameters[parameters.param == 'lambda_'][parameters.covariate == 'Intercept'].coef.values[0]
    covariates = parameters[parameters.param == 'lambda_'].covariate.unique().tolist()
    covariates.remove("Intercept")
    lambdaX = data[covariates].dot(np.array(parameters[parameters.param == 'lambda_'][parameters.covariate != 'Intercept'].coef.tolist())) + almbda_inter
    lambdaX = np.array(exp(lambdaX)).reshape((len(data), 1))
    lambdaX_t = np.array(range(0,len(t),1)).reshape((1,len(t))) / lambdaX
    survival = exp(-np.power(lambdaX_t,rho))
    out = {'time':t, 'survival':survival}
    return(pd.DataFrame(survival))
    
time = range(0,70,1)
test = calculate_survival_fun(X, time, aft)
test


# ### Reference:
# 
# https://towardsdatascience.com/survival-analysis-intuition-implementation-in-python-504fde4fcf8e
# 
# https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
# 
# https://medium.com/analytics-vidhya/concordance-index-72298c11eac7
# 
# https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
