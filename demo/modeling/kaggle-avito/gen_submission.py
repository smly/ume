# -*- coding: utf-8 -*-
import argparse

import scipy.io as sio
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str, required=True)
    p.add_argument('-o', '--output', type=str, required=True)
    return p.parse_args()


def main(args):
    m = sio.loadmat("data/working/feature.gen_y.mat")
    ids = m['ids_test']
    ids = ids.reshape((ids.size))
    df = pd.read_csv(args.input)
    df['id'] = ids
    df.sort('y_pred', ascending=False)[['id']].to_csv(args.output, index=False)


if __name__ == '__main__':
    main(parse_args())
