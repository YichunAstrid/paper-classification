# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import process_string, sample, data_clean, process_venue
from utils import generate_venue_scores, build_venue_co_occurrence_matrix


train_set_path = 'data/train.json'
df = pd.read_json(train_set_path)
df = df.astype(str)
df = process_venue(df)
df_train, df_dev = sample(df=df, column='authorId')
print(f'The shape of df_train is {df_train.shape}')
print(f'The shape of df_dev is {df_dev.shape}')
print('-' * 100)

# # extract content form column `title` and `abstract` to different list.
# train_title = df_train['title'].tolist()
# train_abstract = df_train['abstract'].tolist()
# dev_title = df_dev['title'].tolist()
# dev_abstract = df_dev['abstract'].tolist()
#
# train_title = [process_string(s) for s in train_title]
# train_abstract = [process_string(s) for s in train_abstract]
# dev_title = [process_string(s) for s in dev_title]
# dev_abstract = [process_string(s) for s in dev_abstract]

# clean text
df_train = data_clean(df_train, 'title', 'title')
df_train = data_clean(df_train, 'abstract', 'abstract')
df_dev = data_clean(df_dev, 'title', 'title')
df_dev = data_clean(df_dev, 'abstract', 'abstract')

train_title = df_train['title'].tolist()
train_abstract = df_train['abstract'].tolist()
dev_title = df_dev['title'].tolist()
dev_abstract = df_dev['abstract'].tolist()


# train two vectorizer for `title` and `abstract`
titles = []
titles.extend(train_title)
titles.extend(dev_title)

abstracts = []
abstracts.extend(train_abstract)
abstracts.extend(dev_abstract)

vectorizer = TfidfVectorizer()
vectorizer.fit(titles + abstracts)

# transfer title
label_authorId = df_train['authorId'].tolist()
X_train_title = vectorizer.transform(train_title)
X_train_abstract = vectorizer.transform(train_abstract)
y_train = df_train['authorId']
venue_train = df_train['venue']

# build venue co-occurrence
co_occurrence_prob, unique_venue = build_venue_co_occurrence_matrix(df)

# X_dev_title = vectorizer.transform(dev_title)
# X_dev_abstract = vectorizer.transform(dev_abstract)
y_dev = df_dev['authorId']
venues_dev = df_dev['venue']

max_acc = 0
alpha = 0
for weight in list(np.linspace(0,1,21)):
    preds = []
    for title, abstract, venue in zip(dev_title, dev_abstract, venues_dev):
        X_dev_title = vectorizer.transform([title])
        X_dev_abstract = vectorizer.transform([abstract])

        title_sim_scores = cosine_similarity(X_dev_title, X_train_title)
        abstract_sim_score = cosine_similarity(X_dev_abstract, X_train_abstract)
        total_sim_score = weight * title_sim_scores + (1 - weight) * abstract_sim_score
        venue_scores = generate_venue_scores(venue, venue_train, co_occurrence_prob, unique_venue)
        total_sim_score *= venue_scores
        
        max_score_index = np.argmax(total_sim_score, axis=1)

        pred = label_authorId[max_score_index[0]]
        preds.append(pred)
        
    acc = accuracy_score(y_dev, preds)
    if acc > max_acc:
        max_acc = acc
        alpha = weight
        print(f"Current acc is {acc} and weight is {weight}." )
        
print('-' * 100)
print(f'Best Acc is {max_acc} and weight alpha is {alpha}')
print('-' * 100)

# # inference
# test_set_path = 'data/test.json'
# df_test = pd.read_json(test_set_path)
# df_test = df_test.astype(str)
#
# test_title = df_test['title'].tolist()
# test_abstract = df_test['abstract'].tolist()
# X_test_title = vectorizer4title.transform(test_title)
# X_test_abstract = vectorizer4abstract.transform(test_abstract)
#
# title_sim_score = cosine_similarity(X_test_title, X_train_title)
# abstract_sim_score = cosine_similarity(X_test_abstract, X_train_abstract)
# total_sim_score = alpha * title_sim_score + (1 - alpha) * abstract_sim_score
# max_score_index = np.argmax(total_sim_score, axis=1)
# pred_authodId = [label_authorId[i] for i in max_score_index]
# df_test['authorId'] = pred_authodId
#
# result = df_test[['paperId', 'authorId']]
# result.to_csv('result.csv')
