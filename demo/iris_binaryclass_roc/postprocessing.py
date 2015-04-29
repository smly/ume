# -*- coding: utf-8 -*-
import pandas as pd


def change_classname(output_fn):
    pd.read_csv(output_fn).rename(columns={
        "Proba": "is_setosa",
    }).to_csv(output_fn, index=None)
