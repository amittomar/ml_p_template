import os
from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
from sklearn import metrics
from . import dispatcher
import joblib
import numpy as np

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        #Load model, encoder and colums
        encoders = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl"))
        clf = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl")) 
        #encode the test dataframe
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]
        
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    predictions /= 5
    sub = pd.DataFrame(np.column_stack((test_idx,predictions)),columns=["id","target"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.id = submission.id.astype(int)
    submission.to_csv(f"models/{MODEL}.csv",index=False)
    