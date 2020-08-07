# Disaster Response Pipeline Project

### Project Overview:
In this project, data engineering methods was applied to analyze disaster data and to build a model for an API that classifies disaster messages.

### Files in repository: 

1. a data set which are real messages and corresponding categories: disaster_messages.csv and disaster_categories.csv
2. a ETL pipeline python file: process_data.py and corresponding output processed database by raw dataset: DisasterResponse.db
3. a ML pipeline to do classification: train_classifier.py and corresponding classification result: classifier.pkl
4. a web app displaying visualization: run.py


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
