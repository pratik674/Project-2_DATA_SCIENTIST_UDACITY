#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[1]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine


# In[2]:


# load messages dataset
messages = pd.read_csv('messages.csv')
messages.shape


# In[3]:


messages.info()


# In[4]:


messages.isnull().sum()


# In[5]:


# load categories dataset
categories = pd.read_csv('categories.csv')
categories.shape


# In[6]:


categories.isnull().sum()


# In[7]:


categories.info()


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[8]:


# merge datasets
df = messages.merge(categories, how='inner', on ='id')
df.isnull().sum()


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# In[9]:


# create a dataframe of the 36 individual category columns
categories_df = df['categories'].str.split(';', expand = True)
categories_df.isnull().sum()


# In[10]:


# select the first row of the categories dataframe
row = categories_df.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.astype(str).apply(lambda x: x[:-2]).tolist()
#category_colnames= pd.Series(row).str.split('-').str[0]
print(category_colnames)


# In[11]:


# rename the columns of `categories`
categories_df.columns = category_colnames
categories_df.head()


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[12]:


for column in categories_df:
    # set each value to be the last character of the string
    #categories_df[column] = categories_df[column].str.strip().str[-1]
    categories_df[column] = categories_df[column].astype(str).str[-1]
        
    # convert column from string to numeric
    categories_df[column] = pd.to_numeric(categories_df[column])
categories_df.head()


# In[13]:


categories_df.isnull().sum()


# In[14]:


categories_df.info()


# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.

# In[15]:


# drop the original categories column from `df`
df = df.drop(['categories'], axis = 1 )

df.head()


# In[16]:


df.isnull().sum()


# In[17]:


df['id'].value_counts()


# In[18]:


# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories_df], axis=1)
df.isnull().sum()


# In[19]:


for column in df.columns:
    print(df[column].unique())


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# In[20]:


df['related'] = df['related'].replace(2,1)
df[df['related'] == 2]


# In[21]:


# check number of duplicates
df.duplicated().sum()


# In[22]:


# drop duplicates
df= df.drop_duplicates()


# In[23]:


# check number of duplicates
df.duplicated().sum()


# In[24]:


df.dropna()


# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# In[25]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('InsertTableName', engine, if_exists = 'append', index=False)


# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[ ]:




