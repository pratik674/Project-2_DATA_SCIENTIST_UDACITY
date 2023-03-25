# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:07:00 2023

@author: prati
"""

import sys

# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize,sent_tokenize
#from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(database_filepath):
    
    
    """
    Function to load the data from the database
    Args:
    database_filepath: database to load the data from
    Return:
    X: dataframe containing the messages
    Y: dataframe containing the categories
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'original'], axis=1)
    # Define feature and target variables X and Y
    X = df['message']
    Y = df.drop(['id', 'message', 'genre', 'original'], axis=1)
    return X, Y


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
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3)
    
   

    return pipeline

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
                'ngram_range': [(1,2)],
                'tfidf__use_idf': (True, False),
                'clf__estimator__n_estimators': [50, 100],
                'clf__estimator__min_samples_split': [2, 3],
                'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000],
                'binarize': [0.01, 0.1, 0, 1, 10, 100, None] }

    cv = GridSearchCV(pipeline2, param_grid=parameters, n_jobs=-1, verbose=3)

    return cv



def evaluate_model(model,X_train,Y_train, X_test, Y_test):
    """
    Function to evaluate the model with classification report and accuracy
    Args:
    model: model to evaluate
    X_test: test dataframe with message-data
    Y_test: test dataframe with the target-data
    """
    model = build_model()
    model.fit(X_train,Y_train)
    
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
    Function to evaluate the model with classification report 
    and accuracy
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

def save_model(model, model_filepath):
    """
    Function to save the model as a Pickle-file
    Args:
    model: model to save
    model_filepath: path of the Pickle-file (str)
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model,X_train,Y_train, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        

if __name__ == '__main__':
    main()
