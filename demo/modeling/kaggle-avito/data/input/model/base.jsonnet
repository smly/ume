// Base
{
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
    },
}
