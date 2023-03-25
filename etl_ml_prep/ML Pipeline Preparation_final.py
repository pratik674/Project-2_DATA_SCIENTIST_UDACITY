#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import pickle


# In[2]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)
X = df['message']
Y = df.drop(['id', 'message', 'genre', 'original'], axis=1)


# ### 2. Write a tokenization function to process your text data

# In[3]:


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    """
    Function to split text into words and lemmatize them
    Args:
    text: Textinput (str)
    Return:
    clean_tokens: list of the words from the text
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[4]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[5]:


from datetime import datetime
start_time = datetime.now()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

model = pipeline
model.fit(X_train, Y_train)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[6]:


from datetime import datetime
start_time = datetime.now()

Y0_pred = model.predict(X_test)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[7]:


from sklearn.metrics import classification_report


# In[8]:


from datetime import datetime
start_time = datetime.now()

i = 0
for col in Y_test:
    print('Feature {}: {}'.format(i + 1, col))
    print(classification_report(Y_test[col], Y0_pred[:, i]))
    i = i + 1
accuracy = (Y0_pred == Y_test.values).mean()
print('The model accuracy is {:.3f}'.format(accuracy))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[9]:


from datetime import datetime
start_time = datetime.now()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[10]:


model.get_params()


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[11]:


parameters = {
    'vect__max_features': (None, 5000),
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 3],
}

cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


from datetime import datetime
start_time = datetime.now()

cv.fit(X_train, Y_train)
Y_pred = cv.predict(X_test)
i = 0
for col in Y_test:
    print('Feature {}: {}'.format(i + 1, col))
    print(classification_report(Y_test[col], Y_pred[:, i]))
    i = i + 1
accuracy = (Y_pred == Y_test.values).mean()
print('The model accuracy is {:.3f}'.format(accuracy))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:


def load_data():
    df = pd.read_sql_query('SELECT * FROM InsertTableName', engine.connect())
    # Define feature and target variables X and Y
    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'original'], axis=1)
    return X, Y

def display_results(cv, Y_test, Y_pred):
    labels = np.unique(Y_pred)
    confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (Y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)

def main():
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    model = build_model()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    display_results(model, Y_test, Y_pred)

def build_model():
    """
    Function to build the model in a pipeline with parameter tuning (GridSearcCV)
    Return:
    cv: model built with a pipeline of transformers (CountVectorizser, TfidfTransformer) and multi-output-classifer
    (RandomForestClassifier)
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__max_features': (None, 5000),
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3)

    return cv


def build_model2():
    """
    Function to build the model in a pipeline with parameter tuning (GridSearcCV)
    Return:
    cv: model built with a pipeline of transformers (CountVectorizser, TfidfTransformer) and multi-output-classifer
    (RandomForestClassifier)
    """
    pipeline2 = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(BernoulliNB()))
    ])

    parameters={'vect__max_features': (None, 5000),
                'ngram_range': [(1,1),(1,2),(1,3)],
                'tfidf__use_idf': (True, False),
                'clf__estimator__n_estimators': [50, 100],
                'clf__estimator__min_samples_split': [2, 3],
                'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000],
                'binarize': [0.01, 0.1, 0, 1, 10, 100, None] }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Function to evaluate the model with classification report and accuracy
    Args:
    model: model to evaluate
    X_test: test dataframe with message-data
    Y_test: test dataframe with the target-data
    """
    model = build_model()
    Y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i = i + 1
    accuracy = (Y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))

def evaluate_model2(model, X_test, Y_test):
    """
    Function to evaluate the model with classification report and accuracy
    Args:
    model: model to evaluate
    X_test: test dataframe with message-data
    Y_test: test dataframe with the target-data
    """
    model = build_model2()
    Y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i = i + 1
    accuracy = (Y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


# ### 9. Export your model as a pickle file

# In[ ]:


with open(model_filepath, 'wb') as f:
    pickle.dump(model, f)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




