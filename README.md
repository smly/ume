UME
===

My personal framework/toolbox for a data mining task.

This application's main goal is to provide a framework that is able to increase
reproducibility and productivity for a data mining competition on kaggle.com.

Bootstrapping
-------------

UME requires very strict dependencies, so setting up a new virtual environment for UME is recommended:

```
$ pwd
/home/smly/workspace/ume
$ ./bootstrap
(snip)
$ source venv/bin/active
$ which python
/home/smly/workspace/ume/venv/bin/python
$ which pip
/home/smly/workspace/ume/venv/bin/pip
$ pip install https://github.com/smly/ume/archive/v1.5.8.zip
$ source venv/bin/active
$ which ume
/home/smly/workspace/ume/venv/bin/ume
```

UME v1.5.8 strictly requires following versions:

* Python 3.4.1
* numpy 1.9.0
* scipy 0.14.0
* pandas 0.14.1
* matplotlib 0.14.0
* scikit-learn 0.15.2
* Cython
* jsonnet 0.0.1


Usage
-----

```
$ ls feature.py utils.py
feature.py  utils.py

$ ume feature -n feature -a
$ ume validate -m data/input/model/lr_tfidf.json
$ ume predict -m data/input/model/lr_tfidf.json -o data/working/result/lr_tfidf.csv
$ cat data/input/model/lr_tfidf.json
{
    "model": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "C": 1.0
        }
    },
    "features": [
        "data/working/feature.gen_tfidf.mat"
    ],
    "target": {
        "file": "data/working/feature.gen_y.mat",
        "name": "y"
    },
    "idx": {
        "train": {
            "file": "data/working/feature.gen_y.mat",
            "name": "idx_train"
        },
        "test": {
            "file": "data/working/feature.gen_y.mat",
            "name": "idx_test"
        }
    },
    "metrics": {
        "method": "ume.metrics.apk_score",
        "params": { }
    },
    "prediction": {
        "method": "utils.PredictProba",
        "params": { }
    },
    "cross_validation": {
        "method": "ume.utils.kfoldcv",
        "params": {
            "n_folds": 5
        }
    }
}
```

Documentation
-------------

I have no plan to describe how to use UME in details.
If you are interested in it, plase send me pull request =)
