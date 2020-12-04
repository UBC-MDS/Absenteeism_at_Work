# author: group 21
# date: 2020-11-27

"""A script that loads the training data frame and creates the preprocessor to use in the 
   machine learning pipeline

Usage: preprocessing_machine_learning.py --input_train=<input_train> --out_dir=<out_dir> 

Options:
--input_train=<input_train>          Path (including filename) to training data (feather file) used for the preprocessing
--out_dir=<out_dir>                  Path to directory where the results dataframes should be written
"""

from docopt import docopt
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)


opt = docopt(__doc__)

def main(input_train, out_dir):
  
  # read the train dataframe
  train_df = pd.read_feather(input_train)
  
  # remove the outliner
  mean_hours = train_df["Absenteeism time in hours"].mean()
  sd_hours = train_df["Absenteeism time in hours"].std()
  train_df = train_df[train_df["Absenteeism time in hours"] < mean_hours +3*sd_hours]
  
  # split train_df into X_train, y_train
  X_train, y_train = train_df.drop(columns = ["ID", "Absenteeism time in hours"]), train_df["Absenteeism time in hours"]

  # create the features lists by type
  numeric_features = X_train.select_dtypes('number').columns.tolist()
  binary_features = X_train.select_dtypes('bool').columns.tolist()
  categorical_features = X_train.select_dtypes('category').drop(columns=["Education"]).columns.tolist()
  ordinal_features =['Education']
  education_levels = [1,2,3,4]
  
  # carry out cross validation
  numeric_transformer = make_pipeline(StandardScaler())
  categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
  binary_transformer = make_pipeline(OneHotEncoder(drop="if_binary", dtype=int))
  ordinal_transformer = make_pipeline(OrdinalEncoder(categories=[education_levels], dtype=int))
  
  preprocessor = make_column_transformer(
  (numeric_transformer, numeric_features), 
  (categorical_transformer, categorical_features),
  (binary_transformer, binary_features),
  (ordinal_transformer, ordinal_features))
  
  # save the result in a pickle file
  processor_file = out_dir + "/processor.pickle"

  pickle_out = open(processor_file,"wb")
  pickle.dump(preprocessor,pickle_out)
  pickle_out.close()
  
      
if __name__ == "__main__":
  main(opt["--input_train"], opt["--out_dir"])
  
