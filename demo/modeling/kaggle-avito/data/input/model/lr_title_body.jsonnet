import "base.jsonnet" {
    model: {
        class: "sklearn.linear_model.LogisticRegression",
        params: {
            C: 1.0
        }
    },
    features: [
        "data/working/feature.gen_tfidf.mat",
        "data/working/feature.gen_title.mat",
    ],
}
