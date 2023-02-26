# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from time import time
from utils import process_string, sample, data_clean
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


train_set_path = 'data/train.json'
df = pd.read_json(train_set_path)
df['year'] = df['year'].astype(int)
df = data_clean(df, 'title', 'title')
df = data_clean(df, 'abstract', 'abstract')

df_train, df_dev = sample(df=df, column='authorId')
df_train.reset_index(inplace=True)
df_dev.reset_index(inplace=True)
print(f'The shape of df_train is {df_train.shape}')
print(f'The shape of df_dev is {df_dev.shape}')
print('-' * 100)

train_title = df_train['title'].tolist()
train_abstract = df_train['abstract'].tolist()
dev_title = df_dev['title'].tolist()
dev_abstract = df_dev['abstract'].tolist()
dev_years = df_dev['year'].astype(int).tolist()

# train two vectorizer for `title` and `abstract`
titles = []
titles.extend(train_title)
titles.extend(dev_title)

abstracts = []
abstracts.extend(train_abstract)
abstracts.extend(dev_abstract)

vectorizer = TfidfVectorizer()
vectorizer.fit(titles + abstracts)
X_train_title = vectorizer.transform(df_train['title'].tolist())
X_train_abstract = vectorizer.transform(df_train['abstract'].tolist())
X_dev_titles = vectorizer.transform(dev_title)
X_dev_abstracts = vectorizer.transform(dev_abstract)

verbose = False
max_acc = 0
year_limit = 5
alpha = 0
for weight in list(np.linspace(0, 1, 21)):
    preds = []
    y_dev = df_dev['authorId']
    for i, year in enumerate(dev_years):
        time1 = time()
        upper, lower = year + year_limit, year - year_limit
        df_train_filter = df_train[(df_train['year'] >= lower) & (df_train['year'] <= upper)]
        y_train = df_train_filter['authorId'].tolist()
        filter_list = df_train_filter.index.tolist()

        time2 = time()
        X_train_abstract_filter = X_train_abstract[filter_list]
        X_train_title_filter = X_train_title[filter_list]
        time3 = time()

        X_dev_title = X_dev_titles[i]
        X_dev_abstract = X_dev_abstracts[i]
        time4 = time()

        title_sim_scores = cosine_similarity(X_dev_title, X_train_title_filter)
        abstract_sim_score = cosine_similarity(X_dev_abstract, X_train_abstract_filter)
        total_sim_score = weight * title_sim_scores + (1 - weight) * abstract_sim_score
        max_score_index = np.argmax(total_sim_score, axis=1)
        time5 = time()

        if verbose:
            print(f'Turn {i} -- filter cost {time2 - time1}')
            print(f'Turn {i} -- get filter train representation cost {time3 - time2}')
            print(f'Turn {i} -- get dev representation cost {time4 - time3}')
            print(f'Turn {i} -- cal sim cost {time5 - time4}')

        pred = y_train[max_score_index[0]]
        preds.append(pred)

    acc = accuracy_score(y_dev, preds)
    if acc > max_acc:
        max_acc = acc
        alpha = weight
        print(f"Current acc is {acc} and weight is {weight}.")

print('-' * 100)
print(f'Best Acc is {max_acc} and weight alpha is {alpha}')
print('-' * 100)

# ##############################################################################
# Inference on test set                                                        #
# ##############################################################################
test_set_path = 'data/test.json'
df_test = pd.read_json(test_set_path)
df['year'] = df['year'].astype(int)
df_reference = df
reference_author_id = df_reference['authorId'].tolist()

test_titles = df_test['title'].tolist()
test_abstracts = df_test['abstract'].tolist()
test_years = df_test['year']
assert len(test_years) == len(test_titles)
X_test_titles = vectorizer.transform(test_titles)
X_test_abstracts = vectorizer.transform(test_abstracts)
X_reference_titles = vectorizer.transform(df_reference['title'].tolist())
X_reference_abstracts = vectorizer.transform(df_reference['abstract'].tolist())

# alpha = 0.15
# year_limit = 5
pred_author_id = []
for i, year in enumerate(test_years):
    upper, lower = year + year_limit, year - year_limit
    df_reference_filter = df_reference[(df_reference['year'] >= lower) & (df_reference['year'] <= upper)]

    y_reference = df_reference_filter['authorId'].tolist()
    filter_list = df_reference_filter.index.tolist()
    X_reference_abstract_filter = X_reference_abstracts[filter_list]
    X_reference_title_filter = X_reference_titles[filter_list]

    X_test_title = X_test_titles[i]
    X_test_abstract = X_test_abstracts[i]

    title_sim_scores = cosine_similarity(X_test_title, X_reference_title_filter)
    abstract_sim_score = cosine_similarity(X_test_abstract, X_reference_abstract_filter)
    total_sim_score = alpha * title_sim_scores + (1 - alpha) * abstract_sim_score
    max_score_index = np.argmax(total_sim_score, axis=1)

    pred = reference_author_id[max_score_index[0]]
    pred_author_id.append(pred)
    
df_test['authorId'] = pred_author_id

result = df_test[['paperId', 'authorId']]
result.to_csv('result.csv')