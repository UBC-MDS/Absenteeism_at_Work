# author: Yiki Su
# date: 2020-11-26

"""The sript reads pre-processed training data from the absenteeism data and applies
the dummy regressor to do the cross validation. The results of the cross validation will
be saved in a feather file.

Usage: read_clean_split_data.py --input=<input> --out_dir=<out_dir> 

Options:
--input=<input>          Path (including filename) to training data (which needs to be saved as a feather file)
--out_dir=<out_dir>      Path to directory where the results dataframes should be written
"""
from docopt import docopt
import pandas as pd
import numpy as np
import os
import os.path
import feather
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate

opt = docopt(__doc__)

def main(input, out_dir):
  
  # read the train dataframe
  train_df = pd.read_feather(input)
  
  # split train_df into X_train, y_train
  X_train, y_train = train_df.drop(columns = ["ID", "Absenteeism time in hours"]), train_df["Absenteeism time in hours"]

  # build the dummy regressor model
  dummyre = DummyRegressor()
  
  # carry out cross validation
  result = pd.DataFrame(pd.DataFrame(cross_validate(dummyre, X_train, y_train, return_train_score=True)).mean())
  
  # save the result in a feather file
  dummy_file = out_dir + "/dummy_result.feather"
  try:  
      feather.write_dataframe(result, dummy_file)
  except:
      os.makedirs(os.path.dirname(dummy_file))
      feather.write_dataframe(result, dummy_file)
      
if __name__ == "__main__":
  main(opt["--input"], opt["--out_dir"])
  
