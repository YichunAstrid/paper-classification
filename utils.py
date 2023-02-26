# -*- coding: utf-8 -*-
import re
import json
import pandas as pd
import numpy as np
from pandas import DataFrame
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from nltk import word_tokenize, pos_tag
import spacy
from scipy.special import softmax


# en_core = spacy.load('en_core_web_sm')

# df_train['title_abstract'] = df_train['title_abstract'].apply(lambda x: " ".join([y.lemma_ for y in en_core(x)]))
# df_train['title_abstract_Nadj'] = df_train['title_abstract_Nadj'].apply(lambda x: " ".join([y.lemma_ for y in en_core(x)]))

def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adj)


def process_string(s: str):
    s = s.lower()
    s = re.sub(r'\{|\}|\(|\)|-|_|:', '', s)
    s = ' '.join(s.split())
    
    return s
    
def data_clean(df: DataFrame, column: str, new_column: str):
    df[new_column] = df[column].apply(lambda x: process_string(x))
    # df[new_column] = df[new_column].apply(lambda x: nouns_adj(x))
    # df[new_column] = df[new_column].apply(lambda x: " ".join([y.lemma_ for y in en_core(x)]))
    
    return df

def clean_venue(s):
    if '@' in s:
        return s.split('@')[1]
    return s
    

def process_venue(df: DataFrame):
    df['venue'] = df['venue'].fillna('Unknown')
    df['venue'] = df['venue'].apply(lambda x: clean_venue(x))
    return df

def process_year(df: DataFrame):
    mean_value = df['year'].mean()
    df['year'].fillna(value=mean_value, inplace=True)
    years = df['year'].tolist()
    scaled_years = MinMaxScaler(years)
    df['year'] = scaled_years
    
    return df


def counter(df: DataFrame, column: str):
    couner = Counter(df[column].tolist())
    
    return counter

def sample(df: DataFrame, column: str):
    train_set, train_label = [], []
    dev_set, dev_label = [], []
    counter = Counter(df[column].tolist())
    exist = Counter([])
    
    for _, row in df.iterrows():
        if counter[row[column]] == 1:
            train_set.append(row)
        elif counter[row[column]] == 2:
            if exist[row[column]] < 1:
                train_set.append(row)
            else:
                dev_set.append(row)
        elif counter[row[column]] == 3:
            if exist[row[column]] < 2:
                train_set.append(row)
            else:
                dev_set.append(row)
        else:
            if exist[row[column]] < counter[row[column]] - 1:
                train_set.append(row)
            else:
                dev_set.append(row)
        exist.update([row[column]])
    
    # convert list to dataframe
    train = pd.DataFrame(train_set)
    dev = pd.DataFrame(dev_set)
    train['authorId'] = train['authorId'].astype(str)
    dev['authorId'] = dev['authorId'].astype(str)

    return train, dev

def build_venue_co_occurrence_matrix(df: DataFrame):
    unique_author_ids = df['authorId'].unique()
    unique_venue = df['venue'].unique().tolist()
    
    co_occurrence_matrix = np.zeros(shape=[len(unique_venue), len(unique_venue)])
    for author_id in unique_author_ids:
        venues = df[df['authorId'] == author_id]['venue'].tolist()

        n = len(venues)
        for i in range(n):
            for j in range(i + 1, n):
                index_i = unique_venue.index(venues[i])
                index_j = unique_venue.index(venues[j])
                co_occurrence_matrix[index_i][index_j] += 1
    
    co_occurrence_prob = softmax(co_occurrence_matrix, axis=1)
    
    return co_occurrence_prob, unique_venue


def generate_venue_scores(cur_venue, candidate_venues, co_occurrence_prob, unique_venue):
    venue_scores = []
    cur_venue_index = unique_venue.index(cur_venue)
    venue_co_occurrence_prob = co_occurrence_prob[cur_venue_index]
    for v in candidate_venues:
        ind = unique_venue.index(v)
        venue_scores.append(venue_co_occurrence_prob[ind])
    
    return np.array(venue_scores)

                
                
            