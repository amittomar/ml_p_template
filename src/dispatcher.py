from sklearn import ensemble
MODELS = {
    "randomforest" : ensemble.RandomForestClassifier(n_estimators=200, n_jobs=4, verbose=2),
    "extracttrees" : ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=2,verbose=2),
}