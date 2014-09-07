// Public LB=0.95511, Private LB=0.95426
import "base.jsonnet" {
    model: {
        class: "sklearn.linear_model.LogisticRegression",
        params: {
            C: 1.0
        }
    },
    features: [
        "data/working/feature.gen_tfidf.mat",
    ],
}
