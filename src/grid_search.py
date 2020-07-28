import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_price_classification/train.csv")
    X = df.drop("price_range",axis =1).values
    y= df.price_range.values
    print(y)
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()

    rf = ensemble.RandomForestClassifier(n_jobs=-1)
    classifier = pipeline.Pipeline([
        ("scaling", scl),
        ("pca",pca),
        ("rf", rf)
    ])
    params = {
        "pca__n_components" : np.arange(5,18),
        "rf__n_estimators" : [100, 200, 300, 400],
        "rf__max_depth" : [1, 3, 5, 7],
        "rf__criterion" : ["gini", "entropy"],
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions = params,
        scoring="accuracy",
        verbose=10,
        n_iter=10,
        n_jobs=1,
        cv=5,
    )
    '''
    Grid search code commented 
    '''
    # model = model_selection.GridSearchCV(
    #     estimator=classifier,
    #     param_grid = params,
    #     scoring="accuracy",
    #     verbose=10,
    #     n_jobs=1,
    #     cv=5,
    # )
    model.fit(X,y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
