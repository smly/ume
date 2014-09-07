# coding: utf-8
import sys
import json
import re
import logging as l

import pandas as pd
import numpy as np
import scipy.sparse as ss
import scipy.io as sio
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from ume.feature_extraction import cat


def _load_df():
    df_train = pd.read_csv('./data/input/avito_train.tsv', delimiter="\t")
    df_test = pd.read_csv('./data/input/avito_test.tsv', delimiter="\t")
    return df_train, df_test


def gen_tfidf():
    stopword_list = stopwords.words('russian')
    df_train, df_test = _load_df()
    vec = TfidfVectorizer(
        stop_words=stopword_list,
        min_df=1,
    )
    df = pd.concat([
        df_train['description'].fillna(""),
        df_test['description'].fillna("")
    ]).reset_index()['description']
    vec.fit(df)
    return {'X': vec.transform(df)}


def gen_title():
    df_train, df_test = _load_df()
    vec = TfidfVectorizer(
        min_df=1,
    )
    df = pd.concat([
        df_train['title'].fillna(""),
        df_test['title'].fillna("")
    ]).reset_index()['title']
    vec.fit(df)
    return {'X': vec.transform(df)}


def gen_category():
    df_train, df_test = _load_df()
    df = pd.concat([
        df_train[['category']].fillna(""),
        df_test[['category']].fillna("")
    ]).reset_index()
    return {'X': cat(df, ['category'])}


def gen_subcategory():
    df_train, df_test = _load_df()
    df = pd.concat([
        df_train[['subcategory']].fillna(""),
        df_test[['subcategory']].fillna("")
    ]).reset_index()
    return {'X': cat(df, ['subcategory'])}


def gen_y():
    df_train, df_test = _load_df()

    y_train = np.array(df_train['is_blocked'])
    ids_train = np.array(df_train['itemid'])
    ids_test = np.array(df_test['itemid'])
    idx_train = np.array(range(len(df_train)))
    idx_test = np.array(range(len(df_train), len(df_train) + len(df_test)))

    y_train = y_train.reshape((y_train.size, 1))
    ids_train = ids_train.reshape((ids_train.size, 1))
    ids_test = ids_test.reshape((ids_test.size, 1))
    idx_train = idx_train.reshape((idx_train.size, 1))
    idx_test = idx_test.reshape((idx_test.size, 1))

    return {
        'y': y_train,
        'ids_train': ids_train,
        'ids_test': ids_test,
        'idx_train': idx_train,
        'idx_test': idx_test,
    }


def gen_attrib():
    df = pd.read_csv("./data/working/attrs.csv")
    cols = [v for v in df.columns.tolist() if v != 'itemid']
    X = None
    for i in range(len(cols)):
        target_uniq = df[cols[i]].astype(str).describe()['unique']
        if target_uniq <= 30:
            X_append = cat(df[[cols[i]]].fillna("@NAN@"), [cols[i]])
        else:
            X_append = ss.csc_matrix(
                df['Цвет'].fillna("@NAN@").apply(
                    lambda x: 0 if x == "@NAN@" else 1
                ).reshape((len(df), 1))
            )

        X = X_append if X is None else ss.hstack((X, X_append))
        l.info({
            'i': "{0} of {1}".format(i, len(cols)),
            'curr': X.shape,
            'name': cols[i],
            'shape': X_append.shape,
            'target_uniq': target_uniq
        })

    return {'X': X}
