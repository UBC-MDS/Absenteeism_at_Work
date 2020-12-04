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

  # re-categorize the Reason of absence feature to avoid overfitting
  reason_list = {
    "Reason for absence":
      {0: 'Others',
      1: 'Health related',
      2: 'Health related',
      3: 'Health related',
      4: 'Health related',
      5: 'Health related',
      6: 'Health related',
      7: 'Health related',
      8: 'Health related',
      9: 'Health related',
      10: 'Health related',
      11: 'Health related',
      12: 'Health related',
      13: 'Health related',
      14: 'Health related',
      15: 'Health related',
      16: 'Health related',
      17: 'Health related',
      18: 'Health related',
      19: 'Health related',
      20: 'Health related',
      21: 'Health related',
      22: 'Health related',
      23: 'Health related',
      24: 'Others',
      25: 'Health related',
      26: 'Others',
      27: 'Health related',
      28: 'Health related'}
      }
  X_train = X_train.replace(reason_list)
  
  # create the features lists by type
  numeric_features = X_train.select_dtypes('number').drop(columns=["Body mass index", "Service time"]).columns.tolist()
  binary_features = X_train.select_dtypes('bool').drop(columns=["Disciplinary failure"]).columns.tolist()
  categorical_features = X_train.select_dtypes('category').drop(columns=["Education", "Month of absence"]).columns.tolist()
  ordinal_features =['Education']
  drop_features = ["ID","Disciplinary failure", "Body mass index", "Service time", "Month of absence"]
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
  
