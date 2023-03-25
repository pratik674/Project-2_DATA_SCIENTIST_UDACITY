#  Disaster Response Pipeline Project

## Table of Contents

### 1. Motivation

### 2. File description

### 3. Results

### 4. Instructions & libraries used

### 5. Acknowledgements


#### Motivation
For any given disaster, specially ones in densely populated areas, timely help,,in the form of resources such as food water medicine etc., for the affected people is a necessity. Organizations dedicated to helping and rescuing people are flooded with emergency messages.
Responding to the messages implies redirect the type of help to a specific department. This requires an automated classification of messages such that the concerned department jumps into action as soon as possible.
The current project is committed to  create an engine which classifies these messages into categories such that the departments handling these categories can respond efficiently.


The aim of the project was divided into 3 sections:

1. creating an ETL-pipeline that takes two CSV files, merges and cleans them, and stores the result in a SQLite database.

2. creating a ML-pipeline that takes the data from the database and processes text and performs a multi-output classification. The script uses NLTK, scikit-learn's Pipeline and GridSearchCV.
3. deploying the trained model in a Flask web app where you can input a new message and get classification results in different categories.


#### File description
The data used for this supervised learning project were a dataset of disaster responding data (disaster_messages.csv) and the corresponding category data (disaster_categories.csv). The datasets were provided by Figure Eight.

The following files has been uploaded to the repository:

app

    run.py: python script to run the model for the web app.

    templates

    master.html: script to construct the web app.

    go.html: extension for master.html.


data

    process_data.py: Python script created with the above preparation to create the pipeline.

    disaster_message.csv: data input for the message data.

    disaster_categories.csv: data input for the category data.

    InsertDatabaseName.db: database containing the merged, cleaned data produced by the process_data.py script


models

    train_classifier.py: Python script to create the machine learning pipeline.

    (missing, too big) classifier.pkl: Pickle file that contains the model from the train_classifier.py script.
    preparation

etl_ml_preparation

    ETL Pipeline Preparation_final.ipynb: Jupyter notebook to create the ETL script and get an overview of the data.
    ML Pipeline Preparation.final.ipynb: Jupyter notebook to prepare the machine learning pipeline, create the tokenize function.
    README.md

#### Results

As already mentioned, the result of this project is a web app that can be used to classify events into different categories so it will be possible to forward the message to the appropiate disaster relief agency.

Screenshot of the Disaster Response Pipeline: shared in the main folder 


Reflection
I am satisfied with the result and fully functional web app. Although everything works, there is still enough room for improvements:

Testing different estimators to optimise the classification (RandomForestClassifier() was used in this project. Apart from this BernouilleNB was also tested and its run time is faster than that for RandomForestClassifier. 
Setting an extended list of parameters for GridSearchCV to optimise the model (long performance for RandomForestClassifier ). 



#### Instructions
This instruction were originally made by the team at Udacity's:

            1. Run the following commands in the project's root directory to set up your database and model.

            - To run ETL pipeline that cleans data and stores in database
            `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/InsertDatabaseName.db`
            - To run ML pipeline that trains classifier and saves
            `python models/train_classifier.py data/InsertDatabaseName.db models/classifier.pkl`

            2. Go to `app` directory: `cd app`

            3. Run your web app: `python run.py`

            4. Click the `PREVIEW` button to open the homepage

Scikit learn , nltk, pandas libraries were used to achieve the objective of this project. 

Go to http://0.0.0.0:3000/


Acknowledgments
I would like to thank the team from Udacity's for the detailed outline for this project and Figure Eight for providing the data to work with.
