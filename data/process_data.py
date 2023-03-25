# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:15:01 2023

@author: prati
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load to csv-files and to merge them afterwards
    Args:
    messages_filepath: first csv file
    categories_filepath: second csv file
    Return:
    df: merged dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    """
    Function to clean the dataframe df for the pipeline
    Args:
    df: dataframe
    Return:
    df: cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories_df = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories_df.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.astype(str).apply(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories_df.columns = category_colnames

    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].astype(str).str[-1]

        # convert column from string to numeric
        categories_df[column] = pd.to_numeric(categories_df[column])

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_df], axis=1, join='inner')

    # replace the in column related the entry 2 with 1 (the column now is binary and not multiclass)
    df['related'] = df['related'].replace(2, 1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Function to save the dataframe df in a database
    Args:
    df: dataframe that needs to be saved
    database-filename: the name of the database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, if_exists='replace',index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

