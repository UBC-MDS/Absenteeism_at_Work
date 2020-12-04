# author: group 21
# date: 2020-11-27
"""A script that loads the training and test sets as well as the preprocessor
   object to perform the machine learnng training and predicton
   
Usage: machine_learning_model.py --input_train=<input_train> --input_test=<input_test> --input_processor=<input_processor> --out_dir=<out_dir> 

Options:
--input_train=<input_train>                 Path (including filename) to training data (feather file) used for model training
--input_test=<input_test>                   Path (including filename) to test data (feather file) used for the scoring
--input_processor=<input_processor>         Path (including filename) to prcessor (pickle file) used in the pipeline
--out_dir=<out_dir>                         Path to directory where the results dataframes should be written
"""
from docopt import docopt
import pickle
import numpy as np
import os
import os.path
import feather
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import  make_column_transformer
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.feature_selection import RFE, RFECV
opt = docopt(__doc__)
def main(input_train, input_test, input_processor, out_dir):
  # read the train and test dataframes
  train_df = pd.read_feather(input_train)
  test_df = pd.read_feather(input_test)
  #read the preprocessor
  pickle_in = open(input_processor,"rb")
  preprocessor = pickle.load(pickle_in)
  # split train_df and test_df
  X_train, y_train = train_df.drop(columns = ["ID", "Absenteeism time in hours"]), train_df["Absenteeism time in hours"]
  X_test, y_test = test_df.drop(columns = ["ID", "Absenteeism time in hours"]), test_df["Absenteeism time in hours"]
  numeric_features = X_train.select_dtypes('number').columns.tolist()
  binary_features = X_train.select_dtypes('bool').columns.tolist()
  categorical_features = X_train.select_dtypes('category').drop(columns=["Education"]).columns.tolist()
  ordinal_features =['Education'] 
  # Testing three models with cross-validation
  results_original_dict ={}
  models ={
    "Linear SVM":SVR(kernel="linear"),
    "Ridge":RidgeCV(),
    "Random Forest":RandomForestRegressor()
          }
  for name,model in models.items():
        pipeline = make_pipeline(preprocessor, model)
        results_original_dict[name] = pd.DataFrame(cross_validate(pipeline, X_train, y_train, cv=5, return_train_score=True, n_jobs=-1)).mean()
  non_RFE_results = pd.DataFrame(results_original_dict).reset_index()
  #Saving the results without RFE  
  non_RFE_results_file = out_dir + "/non_RFE_CV_results.feather"
  try:  
      feather.write_dataframe(non_RFE_results, non_RFE_results_file)
  except:
      os.makedirs(os.path.dirname(non_RFE_results_file))
      feather.write_dataframe(non_RFE_results, non_RFE_results_file)
  #Testing same models with recursive feature elimination
  results_dict ={}
  models ={
    "Linear SVM":SVR(kernel="linear"),
    "Ridge":RidgeCV(),
    "Random Forest":RandomForestRegressor()
          }
  for name,model in models.items():
        pipeline = make_pipeline(preprocessor,RFECV(Ridge(), cv=5), model)
        results_dict[name] = pd.DataFrame(cross_validate(pipeline, X_train, y_train, cv=5, return_train_score=True, n_jobs=-1)).mean()      
  RFE_results = pd.DataFrame(results_dict).reset_index()
  #Saving the results with RFE  
  RFE_results_file = out_dir + "/RFE_CV_results.feather" 
  feather.write_dataframe(RFE_results, RFE_results_file)
  #Ridge selection afterselecting a model (feature selection considered)
  ridge_pipeline = make_pipeline(preprocessor,RFECV(Ridge(), cv=5), RidgeCV())
  ridge_pipeline.fit(X_train, y_train)
  #getting most influential attributes
  total_features = (numeric_features + list(preprocessor.
                             named_transformers_["pipeline-2"].
                             named_steps["onehotencoder"].
                             get_feature_names(categorical_features)) + 
                    binary_features + ordinal_features)
  lr_coefs = ridge_pipeline[2].coef_
  attributes= pd.Series(total_features)[ridge_pipeline.named_steps["rfecv"].support_]
  best_attributes=pd.DataFrame(data=lr_coefs, index=attributes.values, columns=["Coefficients"])
  best_attributes = best_attributes.reset_index()
  #Saving the coefficients and attributes
  coef_path = out_dir + "/best_coefficients.feather" 
  feather.write_dataframe(best_attributes, coef_path)
  #Scoring the model
  score = ridge_pipeline.score(X_test, y_test)
  print(f"The R2 test score obtained is {round(score,3)}")
  #Saving score in a pickle file  
  test_score_file = out_dir + "/test_score.pickle"
  pickle_out = open(test_score_file,"wb")
  pickle.dump(score, pickle_out)
  pickle_out.close()
if __name__ == "__main__":
  main(opt["--input_train"], opt["--input_test"], opt["--input_processor"], opt["--out_dir"])
  
