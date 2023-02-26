# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils import process_string, sample
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_similarity(example_repre, corpur_repre) :
    sim_scores = cosine_similarity(example_repre, corpur_repre)
    max_score_index= np.argmax(sim_scores, axis=1)
    
    return max_score_index

train_set_path = 'data/train.json'
test_set_path = 'data/test.json'

df = pd.read_json(train_set_path)
df = df.astype(str)
df_train, df_dev = sample(df=df, column='authorId')
print(f'The shape of df_train is {df_train.shape}')
print(f'The shape of df_dev is {df_dev.shape}')


df_train = pd.read_csv('train.csv')
df_dev = pd.read_csv('dev.csv')

corpus1 = df_train['title'] + " " + df_train['abstract']
corpus2 = df_dev['title'] + " " + df_dev['abstract']

corpus1 = [process_string(s) for s in corpus1.tolist()]
corpus2 = [process_string(s) for s in corpus2.tolist()]
corpus = []
corpus.extend(corpus1)
corpus.extend(corpus2)

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus1)

X_train = vectorizer.transform(corpus1)
y_train = df_train['authorId']
X_dev = vectorizer.transform(corpus2)
y_dev = df_dev['authorId']

sim_scores = cosine_similarity(X_dev, X_train)
max_score_index = np.argmax(sim_scores, axis=1)
max_score_index = calculate_similarity(X_dev, X_train)
label_values = df_train['authorId'].tolist()
pred = [label_values[i] for i in max_score_index]
acc = accuracy_score(y_dev, pred)
print(acc)

