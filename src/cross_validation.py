import pandas as pd
from sklearn import model_selection
"""
- binary classification
- multi class classificatin
- multi label classification
- single column regression
- multi column regression
- holdout
"""

class CrossValidations:
    def __init__(
        self, 
        df, 
        target_cols, 
        problem_type = "binary_classification",
        num_folds = 5, 
        shuffle = True,
        random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_target = len(target_cols)
        self.problem_type = problem_type
        self.num_folds=num_folds
        self.shuffle=shuffle
        self.random_state = random_state


        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac =1).reset_index(drop=True)
        # add the new column kfold to datframe and initiliaze    
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ["binary_classification", "multi_class_classification"]:
            # input validation check
            if self.num_target != 1:
                raise Exception("Invalid number of target for this problem type")
            unique_value = self.dataframe["target"].nunique()
            if unique_value == 1:
                raise Exception("Only one unique value found")

            # Using Stratified Fold for binary and multi class classificatiomn 
            elif unique_value > 1:
                target = self.target_cols[0]
                kf=model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False, random_state=self.random_state)
                
            for fold,(train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe.target.values)):
                self.dataframe.loc[val_idx, 'kfold'] =fold
            
        elif self.problem_type in  ["single_col_regression","multi_column_regression"]:
            # input validation
            if self.num_target != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of target for this problem type")
            if self.num_target < 2 and self.problem_type == "multi_column_regression":
                raise Exception("Invalid number of target for this problem type")
            
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold,(train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] =fold
        else:
            raise Exception("Problem type not understandable")
        
        return self.dataframe

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    cv = CrossValidations(df, target_cols=["target"])
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())