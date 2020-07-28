import pandas as pd
import numpy as np


from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline
from functools import partial
from sklearn.utils.optimize import space
from skopt import space
from skopt import gp_minimize
from sklearn import ensemble 

def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accurecies = []
    for idx in kf.split(X=x, Y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)
        accurecies.append(fold_acc)

    return -1.0 * np.mean(accurecies)



if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_price_classification/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    param_space = [
        space.Integer(low=3, high=15, name="max_depth"),
        space.Integer(low=100, high=600, name="n_estimator"),
        space.Categorical(categories=["gini", "entropy"], name="criterion"),
        space.Real(low=0.01, high=1, prior="uniform"),
    ]
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features",
    ]

    optimization_function = partial(
        optimize,
        param_names=param_names,
        x=X,
        y=y,
    )

    result = gp_minimize(optimization_function,
                         dimensions=param_space,
                         n_calls=10,
                         verbose=10,
                         )
    print(result)
