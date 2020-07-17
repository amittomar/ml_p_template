from sklearn import preprocessing


class CategoricalFeatures:
    def __init__(self,
                 df,
                 categorical_features,
                 encoding_type,
                 handle_na=True):
        """

        df:pandas dataframe
        categorical_features:  list of clumn names , e.g. ["ord_1", "nom_0"................]
        encoding_type : label, binary, one
        """
        self.df = df
        self.out_df = self.df.copy(deep=True)
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-99999999999")

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.out_df.loc[:, c] = lbl.transform(self.df[c].values)
        self.label_encoders[c] = lbl
        return self.out_df
    
    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(df[c].values)
            val = lbl.transform(self.df[c].values)
            self.out_df = self.out_df.drop(c, axis =1)
            
            for j in range(val.shape[1]):
                new_coln_name = c + f"__bin_{j}"
                self.out_df[new_coln_name] = val[:, j]
            
        self.binary_encoders[c] = lbl
        return self.out_df

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe, handle_na=True):
        if handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-99999999999")
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c,axis = 1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

            



if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../input/catagorical_encoding/train.csv").head(500)
    print(df.head())
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats = CategoricalFeatures(df,
                                    categorical_features=cols,
                                    encoding_type="binary",
                                    handle_na=True)
    output_df = cat_feats.fit_transform()
    print("*****************Out put ***************")
    print(output_df.head())
