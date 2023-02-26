# -*- coding: utf-8 -*-

"""
@project: 001-paper_classification
@author: heibai
@file: data_explore.py
@ide: PyCharm
@time 2022/12/8 20:41
"""
import pandas as pd

df = pd.read_json('data/train.json')
author_ids = df['authorId'].unique().tolist()
res = []
for author_id in author_ids:
    years = df[df['authorId'] == author_id]['year'].tolist()
    years = [str(y) for y in years]
    res.append(years)

with open('years_distribution.txt', 'w', encoding='utf-8') as f:
    for r in res:
        f.write(' '.join(r) + '\n')



# df_test = pd.read_json('data/test.json')
# print(df_test['year'].value_counts())