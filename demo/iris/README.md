# Demo using iris dataset

## Model definition

The model in ume framework is defined with jsonnet.

### task specification file (data/input/model/spec.jn):

```json
/*
  Task description
  ----------------
  Title: iris dataset demo
  Metrics: Multi Logloss
  Task: Multi-class classification
*/
{
    "task": {
        "class": "ume.task.MultiClassPredictProba",
        "params": {
            "metrics": {
                "method": "ume.metrics.multi_logloss",
            },
            "validation": {
                "class": "utils.KFold",
                "params": {
                    "n_jobs": 1,
                    "shuffle": "True",
                },
            },
        },
        "dataset": {
            "y_train": { "file": "data/working/feat.gen_first_two.npz", "name": "y" },
            "id_train": { "file": "data/working/feat.gen_first_two.npz", "name": "id_train" },
            "id_test": { "file": "data/working/feat.gen_first_two.npz", "name": "id_test" },
        },
    },
}
```

### model parameter file (data/input/model/v3_linear_model_plus_pca.jn):

```
import "spec.jn" {
    "model": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "C": 10.0
        }
    },
    "features": [
        "data/working/feat.gen_first_two.npz",
        "data/working/feat.gen_pca.npz",
    ],
}
```

## Feature extraction

```bash
$ ume feature -a -n feat
feat
2015-04-25 14:28:00,644 Feature generation: feat.gen_pca
2015-04-25 14:28:00,655 Feature generation: feat.gen_first_two
```

## Cross validation

```bash
$ ume validate -m data/input/model/v3_linear_model_plus_pca.jn
2015-04-25 14:33:38,843 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,844 KFold: (0) 0.4520
2015-04-25 14:33:38,844 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,846 KFold: (1) 0.2612
2015-04-25 14:33:38,846 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,847 KFold: (2) 0.2072
2015-04-25 14:33:38,847 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,849 KFold: (3) 0.1603
2015-04-25 14:33:38,849 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,850 KFold: (4) 0.2232
2015-04-25 14:33:38,850 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,852 KFold: (5) 0.1583
2015-04-25 14:33:38,852 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,853 KFold: (6) 0.2415
2015-04-25 14:33:38,853 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,854 KFold: (7) 0.2498
2015-04-25 14:33:38,855 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,856 KFold: (8) 0.1843
2015-04-25 14:33:38,856 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (90, 5)
2015-04-25 14:33:38,857 KFold: (9) 0.2138
2015-04-25 14:33:38,858 CV Score: 0.2352 (var: 0.006345)
```

## Generate a submission file

```bash
$ ume predict -m data/input/model/v3_linear_model_plus_pca.jn
2015-04-25 14:35:19,599 Clf: LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001), X: (100, 5)

$ head data/output/v3_linear_model_plus_pca.jn.csv
Id,Class_1,Class_2,Class_3
Line92,0.0026394938366868593,0.9938190645282173,0.0035414416350957995
Line28,0.8975356911214508,0.1024643087765222,1.0202702293242527e-10
Line78,0.0004463584219969818,0.8518054446852529,0.14774819689275007
Line53,0.00122228834227828,0.9860657367155592,0.01271197494216249
Line38,0.8357776503310458,0.16422234949645645,1.7249778244375177e-10
Line126,3.202397275500544e-05,0.4721884174596623,0.5277795585675827
Line115,1.5572018398520943e-05,0.12336520416128324,0.8766192238203183
Line43,0.9569647282863225,0.04303527080272157,9.109559142150738e-10
Line36,0.9151981984879709,0.08480180148930301,2.2726024401203313e-11
```

