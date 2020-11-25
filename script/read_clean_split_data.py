# author: Yiki Su
# date: 2020-11-24

"""The sript reads the csv file from the data folder and performs data wrangling.
After data wrangling, separate the data into the training and test portions using 
a 70 : 30 train test split.
Usage: read_clean_split_data.py --input=<input> --out_dir_train=<out_dir_train> --out_dir_test=<out_dir_test>

Options:
--input=<input>                      The path of the raw data csv file
--out_dir_train=<out_dir_train>      Path to directory where the processed train dataframe should be written
--out_dir_test=<out_dir_test>        Path to directory where the processed test dataframe should be written
"""

from docopt import docopt
import pandas as pd
import numpy as np
import os
import os.path
import feather
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(input, out_dir_train, out_dir_test):
  # read the data
  data = pd.read_csv(input, sep=";")
  
  # data wrangling
  data['Social drinker'] = data['Social drinker'].astype('bool')
  data['Social smoker'] = data['Social smoker'].astype('bool')
  data['Disciplinary failure'] = data['Disciplinary failure'].astype('bool')
  data['Seasons'] = data['Seasons'].astype('category')
  data['Education'] = data['Education'].astype('category')
  data['Month of absence'] = data['Month of absence'].astype('category')
  data['Reason for absence'] = data['Reason for absence'].astype('category')
  
  # split the data
  train_df, test_df = train_test_split(data, test_size=0.3, random_state=123)
  
  # save the train_df and test_df to feather files
  try:  
      feather.write_dataframe(train_df, out_dir_train)
  except:
      os.makedirs(os.path.dirname(out_dir_train))
      feather.write_dataframe(train_df, out_dir_train)
  
  try:  
      feather.write_dataframe(test_df, out_dir_test)
  except:
      os.makedirs(os.path.dirname(out_dir_test))
      feather.write_dataframe(test_df, out_dir_test)


if __name__ == "__main__":
  main(opt["--input"], opt["--out_dir_train"], opt["--out_dir_test"])
