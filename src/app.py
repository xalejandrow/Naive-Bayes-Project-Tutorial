import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import pickle

#0.
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')
#1. Transform
df_transf = df_raw.copy()
df_transf = df_transf.drop('package_name', axis=1)
df_transf['review'] = df_transf['review'].str.strip()
df_transf['review'] = df_transf['review'].str.lower()
#2. Split 
df = df_transf.copy()
X = df['review']
y = df['polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=25)
#3. Preproccesing and model MB
clf_3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
clf_3.fit(X_train, y_train)
#4.Randomized search to select hyperparameters
n_iter_search = 5
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf_3 = RandomizedSearchCV(clf_3, parameters, n_iter = n_iter_search)
gs_clf_3.fit(X_train, y_train)
best_model = gs_clf_3.best_estimator_
#5. Save best model
pickle.dump(best_model, open('../models/best_model.pickle', 'wb'))