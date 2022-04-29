#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[197]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[198]:


df = pd.read_csv("ab_data.csv")
df.head(10)


# b. Use the below cell to find the number of rows in the dataset.

# In[199]:


df.shape


# c. The number of unique users in the dataset.

# In[200]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[201]:


df.converted.mean()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[202]:


df.loc[(df.group == "treatment")&(df.landing_page != "new_page")].shape[0]


# f. Do any of the rows have missing values?

# In[203]:


df.isnull().sum()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[204]:


df = df.drop(index = df.loc[(df.group == "treatment")&(df.landing_page != "new_page")].index )
df = df.drop(index = df.loc[(df.group == "control")&(df.landing_page != "old_page")].index)
df2 = df


# In[205]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
df2[((df2['group'] == 'control') == (df2['landing_page'] == 'old_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[206]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[207]:


df2.user_id.value_counts()


# c. What is the row information for the repeat **user_id**? 

# In[208]:


df2.loc[df2.user_id == 773192 ]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[209]:


df2.drop(index = 1899 , inplace= True)


# In[210]:


# test for duplicates 
df2.user_id.duplicated().sum()


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[211]:


df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[212]:


df2.query("group == 'control'")["converted"].mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[213]:


df2.query("group == 'treatment'")["converted"].mean()


# d. What is the probability that an individual received the new page?

# In[214]:


df2.query("landing_page == 'new_page'").shape[0]/df2.shape[0]


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# #**Your answer goes here.**
# There is no sufficient evidence to say that the new treatment page leads to more conversions. 
# The reason is that this is only single sample and can not represent an evidence for all the population  

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# # **Put your answer here.**
# H0 : Pnew - Pold =< 0    
# 
# H1 : Pnew - Pold > 0

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[265]:


P_new = df2.query("converted == 1 ").shape[0]/df2.shape[0]
P_new


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[266]:


P_old = df2.query("converted == 1 ").shape[0]/df2.shape[0]
P_old


# c. What is $n_{new}$?

# In[267]:


n_new = df2.query("group == 'treatment'").shape[0]
n_new


# d. What is $n_{old}$?

# In[268]:


n_old = df2.query("group == 'control'").shape[0]
n_old 


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[338]:


new_page_converted = np.random.choice([0,1] , p = [1-P_new,P_new] , size = n_new).mean()
new_page_converted


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[339]:


old_page_converted = np.random.choice([0,1] , p = [1-P_old,P_old] , size = n_old).mean()
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[340]:


obs_diff = new_page_converted - old_page_converted
obs_diff


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[341]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.choice([0,1] , p = [1-P_new,P_new] , size = n_new)
    old_page_converted = np.random.choice([0,1] , p = [1-P_old,P_old] , size = n_old)
    p_diffs.append(new_page_converted.mean()- old_page_converted.mean())
    


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[342]:


diffs_std = np.std(p_diffs)
null_val = np.random.normal(0,diffs_std, len(p_diffs));
plt.hist(null_val);
plt.axvline(x = obs_diff , c = "r" );


# In[343]:


(null_val>obs_diff).mean()


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[344]:


(p_diffs>obs_diff).mean()


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# # **Put your answer here.**
# this value called P-Value . A p value is used in hypothesis testing to help to support or reject the null hypothesis . The smaller the p-value, the stronger the evidence that We should reject the null hypothesis and in the previous case P-value was more than alpha which is 5% . so We fail to reject the null (the old page) .

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[281]:


df2.head()


# In[296]:


import statsmodels.api as sm

convert_old =  df2.query("landing_page == 'old_page' & converted == 1 ").shape[0]
convert_new =  df2.query("landing_page == 'new_page' & converted == 1 ").shape[0]
n_old = df2.query("landing_page == 'old_page' ").shape[0]
n_new = df2.query("landing_page == 'new_page' ").shape[0]
print( "convert_old" , convert_old)
print( "convert_new" , convert_new)
print ( "n_old" , n_old)
print("n_new" , n_new )


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[298]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')

print(c, p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# # **Put your answer here.**
# p-value is very high so that means We can't  reject the null hypothesis and this support our findings in parts j.
# 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# # **Put your answer here.**
# Logistic regression

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[326]:


df2[["control","ab_page"]] = pd.get_dummies(df2.group)
df2["intercept"] = 1 
df2.head()


# In[327]:


df2.drop(columns = ["control"], inplace = True )


# c. Use statsmodels to import your regression model. Instantiate the model, and fit the model using the two columns you created in part b. to predict whether or not an individual converts.

# In[334]:


log = sm.Logit(df2.converted,df2 [["ab_page" , "intercept"]])
results = log.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[335]:


results.summary()


# In[337]:


1/np.exp(-0.0150 ) , 1/np.exp(-1.9888 )


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# # **Put your answer here.**
# from logistic regression model We can say that user is 1.015 times as more likely to convert when will be in treatment group  receiving  the new page than control group who receive the old page.
# P- value is 0.189 much more than the 0.05  So we fail to reject the H0 that was :  Pnew - Pold =< 0 because we don't have a significant statics evidence .

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# # **Put your answer here.**
# I think it is a good idea to consider other factors to add into your regression model like the age , gender or the incomes to check the relationship between these factors and the conversion rate .
# regarding the disadventages of adding additional terms into the regression model maybe existing of outliers , correlated errors or existing of correlations between the x variables 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[345]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# In[346]:


df_new.head()


# In[347]:


df_new.country.value_counts()


# In[358]:


### Create the necessary dummy variables
df_new[["CA","UK","US"]] = pd.get_dummies(df_new.country)
df_new.head()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[359]:


### Fit Your Linear Model And Obtain the Results
log = sm.Logit(df_new.converted , df_new[["ab_page", "intercept", "UK" , "CA"]])
results_new = log.fit()
results_new.summary()


# In[361]:


np.exp(results_new.params)


# In[363]:


# compute the reciprucal of Canada 
1/np.exp(-0.0408)


#  H0 : there is no significant effects on conversion when we look at an interaction between page and country 
#  
# From Summary We can get the below conclusion that
# 
# P value of canada : 0.129 // P Value  UK : 0.458  . So We fail to  reject the H0
# 
# in UK user likely to conversion 1.009 times of USA 
# 
# in USA user likely to conversion 1.0416 times of Canada 
# 
# There is no strong evidence that the countries (US, CA and UK) influence the conversion rate

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by beginning the next module in the program.
