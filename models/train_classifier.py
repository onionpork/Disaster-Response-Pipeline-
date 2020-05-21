import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib 


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('disaster_cat', engine) 
    X = df['message']
    Y = df.drop(['id','message','original', 'genre'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier())) ])

#     parameters = {
#     'vect__max_df': (0.5, 0.75, 1.0),
#     'tfidf__use_idf': (True, False),
#     'vect__ngram_range': ((1, 1), (1, 2)),
#     'clf__estimator__min_samples_split':[2,3,4],
#     'clf__estimator__n_estimators':[10,20,50]
#     }
    
#     cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    f1_score, precision, recall = [], [],[]

    for i in range(len(category_names)):
        res = classification_report(Y_test.iloc[:,i], y_pred[:,i])
        f1_score.append(float(res.split()[-2]))
        recall.append(float(res.split()[-3]))
        precision.append(float(res.split()[-4]))
    accuracy = (y_pred == Y_test).mean()
    print('accuracy: ', accuracy.mean())
        

def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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