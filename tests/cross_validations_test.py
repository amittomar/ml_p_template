import unittest

import pandas as pd

from src import cross_validation
# from unittest import runner


class CrossValidations(unittest.TestCase):
    def setUp(self):
        #self.binaryClassificationValidator = cross_validation.CrossValidations()
        print("init done")

    def test_binary_cross_validator(self):
        df = pd.read_csv("input/train.csv")
        cv = cross_validation.CrossValidations(df, target_cols=["target"])
        df_split = cv.split()
        #print(df_split.head())
        print(df_split.kfold.value_counts())



if __name__ == "__main__":
    unittest.main()
