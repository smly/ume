# -*- coding: utf-8 -*-
import pandas as pd


def change_classname(output_fn):
    pd.read_csv(output_fn).rename(columns={
        "Class_1": "setosa",
        "Class_2": "versicolor",
        "Class_3": "virginica",
    }).to_csv(output_fn, index=None)
