# demo/modeling/kaggle-avito

## Prepare dataset

```
$ ln -s ~/Downloads/avito_train.tsv data/input/avito_train.tsv
$ ln -s ~/Downloads/avito_test.tsv data/input/avito_test.tsv
```

## Feature generation

```
$ ume feature --name features --all
2014-09-07 21:08:55,774 Feature generation: features.gen_bigram_title
(snip)
```

## Validation

```
$ ume validate -m data/input/model/lr_tfidf.jsonnet
2014-09-07 21:12:20,927 Loading dataset
2014-09-07 21:12:20,927 data/working/feature.gen_tfidf.mat
2014-09-07 21:14:12,544 Training model: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
2014-09-07 21:16:36,615 Training model ... done
2014-09-07 21:16:56,213 Score: 0.9725
2014-09-07 21:17:42,036 Training model: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
2014-09-07 21:20:36,067 Training model ... done
2014-09-07 21:20:55,338 Score: 0.9736
2014-09-07 21:21:39,178 Training model: LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
(snip)
```

## Generate a submission file

```
$ ume predict -m data/input/model/lr_tfidf.jsonnet -o data/working/result/lr_tfidf.csv
2014-09-07 21:23:44,032 Loading dataset
2014-09-07 21:23:44,032 data/working/feature.gen_tfidf.mat
(snip)

$ python gen_submission.py -i data/working/result/lr_tfidf.csv -o data/output/lr_tfidf.csv
```
