# author: Yiki Su
# date: 2020-11-25

"""The sript takes in the train dataframe and performs explanatory data analysis to it.
All output plots will be saved in the figure folder.

Usage: eda.py --input=<input> --out_dir=<out_dir>

Options:
--input=<input>          The path of the train dataframe
--out_dir=<out_dir>      Path to directory where the generated plot should be written
"""

from docopt import docopt
import pandas as pd
import numpy as np
import os
import os.path
import feather
import altair as alt


opt = docopt(__doc__)

def main(input, out_dir):
  
  # read in the train dataframe
  train_df = pd.read_feather(input)
  
  # generate the info table of the train dataframe
  info = train_df.info()
  
  # save the info table to a feather file
  # info_file = out_dir + "/info_df.feather"
  # try:  
  #     feather.write_dataframe(info, info_file)
  # except:
  #     os.makedirs(os.path.dirname(info_file))
  #     feather.write_dataframe(info, info_file)
      
  # generate the correlation matrix
  train_df_copy = train_df.copy()
  train_df_copy['Social drinker'] = train_df_copy['Social drinker'].astype('int64')
  train_df_copy['Social smoker'] = train_df_copy['Social smoker'].astype('int64')
  train_df_copy['Disciplinary failure'] = train_df_copy['Disciplinary failure'].astype('int64')
  train_df_copy['Seasons'] = train_df_copy['Seasons'].astype('int64')
  train_df_copy['Education'] = train_df_copy['Education'].astype('int64')
  train_df_copy['Month of absence'] = train_df_copy['Month of absence'].astype('int64')
  train_df_copy['Reason for absence'] = train_df_copy['Reason for absence'].astype('int64')
  
  corr_df = train_df_copy.corr().stack().reset_index(name='corr')
  corr_matrix = alt.Chart(corr_df).mark_rect().encode(
      x=alt.X('level_0', title=None),
      y=alt.Y('level_1', title=None),
      color=alt.Color('corr', scale=alt.Scale(domain=(-1, 1), scheme='purpleorange'))
  ).properties(width=600, height=600)
  
  corr_fig = out_dir + "/correlation_matrix.png"
  try:  
      feather.write_dataframe(corr_matrix, corr_fig)
  except:
      os.makedirs(os.path.dirname(corr_fig))
      feather.write_dataframe(corr_matrix, corr_fig)


if __name__ == "__main__":
  main(opt["--input"], opt["--out_dir"])
