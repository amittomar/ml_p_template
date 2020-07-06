export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv
# randomforest , extracttrees
export MODEL=$1

python -m src.predict