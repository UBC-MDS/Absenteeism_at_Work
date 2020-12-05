# author: group 21
# date: 2020-12-04
"""A script that loads the training and test sets as well as the preprocessor
   object to perform the machine learnng training and predicton
   
Usage: machine_learning_model.py --input_xtrain=<input_xtrain> --input_ytrain=<input_ytrain> --input_xtest=<input_xtest> --input_ytest=<input_ytest> --input_processor=<input_processor>  --input_total_features=<input_total_features> --out_dir=<out_dir> 

Options:
--input_xtrain=<input_xtrain>                   Path (including filename) to training x data (feather file) used for model training
--input_ytrain=<input_ytrain>                   Path (including filename) to training y data (feather file) used for model training
--input_xtest=<input_xtest>                     Path (including filename) to x test data (feather file) used for the scoring
--input_ytest=<input_ytest>                     Path (including filename) to y test data (feather file) used for the scoring
--input_processor=<input_processor>             Path (including filename) to processor (pickle file) used in the pipeline
--input_total_features=<input_total_features>   Path (including filename) to feature names list (pickle file) used to obtain most important coefficients
--out_dir=<out_dir>                             Path to directory where the results dataframes should be written
"""

from docopt import docopt
import pickle
import seaborn as sns
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.feature_selection import RFE, RFECV
opt = docopt(__doc__)
def main(input_xtrain, input_ytrain, input_xtest, input_ytest, input_processor, input_total_features, out_dir):
  
  # read dataframes
  xtrain_in = open(input_xtrain, "rb")
  X_train = pickle.load(xtrain_in)
  
  ytrain_in = open(input_ytrain, "rb")
  y_train = pickle.load(ytrain_in)
  
  xtest_in = open(input_xtest, "rb")
  X_test = pickle.load(xtest_in)
  
  ytest_in = open(input_ytest, "rb")
  y_test = pickle.load(ytest_in)
  
  #read the preprocessor
  pickle_in = open(input_processor,"rb")
  preprocessor = pickle.load(pickle_in)
  
  features_in = open(input_total_features,"rb")
  total_features = pickle.load(features_in)

  new_index = ["fit_time", "score_time", "validation_r2", "train_r2", "validation_neg_root_mean_square_error", "train_neg_root_mean_square_error"]
  
  
  # Testing three models with cross-validation
  scoring={
    "r2": "r2",
    "neg_root_mean_square_error": "neg_root_mean_squared_error",           
    }

  results_original_dict ={}
  models ={
    "Linear SVM":SVR(kernel="linear"),
    "Ridge":RidgeCV(),
    "Random Forest":RandomForestRegressor(random_state=123)
    }
  for name,model in models.items():
    pipeline = make_pipeline(preprocessor, model)
    results_original_dict[name] = pd.DataFrame(cross_validate(pipeline, X_train, y_train, cv=5, return_train_score=True, n_jobs=-1, scoring=scoring)).mean()  
  original_results=pd.DataFrame(results_original_dict)
  original_results.index = new_index
  non_RFE_results = original_results.reset_index()
  
  
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
    "Random Forest":RandomForestRegressor(random_state=123)
     }
  for name,model in models.items():
      pipeline = make_pipeline(preprocessor,RFECV(Ridge(), cv=5), model)
      results_dict[name] = pd.DataFrame(cross_validate(pipeline, X_train, y_train, cv=5, return_train_score=True, n_jobs=-1, scoring=scoring)).mean()     
  feature_selection_results = pd.DataFrame(results_dict)
  feature_selection_results.index = new_index
  RFE_results = feature_selection_results.reset_index()
  
  
  #Saving the results with RFE  
  RFE_results_file = out_dir + "/RFE_CV_results.feather" 
  feather.write_dataframe(RFE_results, RFE_results_file)
  
  
  # Linear SVM hyperparameter optimization
  SVM_pipeline = make_pipeline(preprocessor,RFECV(Ridge(), cv=5), SVR(kernel="linear"))
  param_grid = {"svr__gamma": 10.0 ** np.arange(-3, 3),
                "svr__C": 10.0 ** np.arange(-3, 3) }
  random_grid = RandomizedSearchCV(SVM_pipeline, param_grid, cv=5, n_jobs=-1, n_iter=10, return_train_score=True, scoring="neg_root_mean_squared_error")
  random_grid.fit(X_train, y_train);
  print("Best cv score from random search: %.4f" % random_grid.best_score_)
  random_grid.best_params_
  random_grid.best_estimator_.fit(X_train, y_train)
  
  
  #getting most influential attributes
  new_colnames = ["Reason for absence_Unknown", 
  "Reason for absence_Physiotherapy",
  "Reason for absence_Medical consultation",
  "Reason for absence_Dental consultation",
  "Reason for absence_Laboratory examination",
  "Reason for absence_Certain conditions originating in the perinatal period",
  "Reason for absence_Neoplasms",
  "Reason for absence_Endocrine, nutritional and metabolic diseases",
  "Social drinker",
  "Reason for absence_Congenital malformations, deformations and chromosomal abnormalities",
  "Reason for absence_Certain infectious and parasitic diseases",
  "Reason for absence_Injury, poisoning and certain other consequences of external causes",
  "Day of the week_Tuesday",
  "Reason for absence_Diseases of the musculoskeletal system and connective tissue",
  "Reason for absence_Diseases of the skin and subcutaneous tissue",
  "Reason for absence_Diseases of the digestive system",
  "Reason for absence_Diseases of the eye and adnexa",
  "Reason for absence_Diseases of the genitourinary system"]
  
  lr_coefs = random_grid.best_estimator_[2].coef_.toarray()
  attributes= pd.Series(total_features)[random_grid.best_estimator_.named_steps["rfecv"].support_]
  best_attributes=pd.DataFrame(data=lr_coefs.transpose(), index=new_colnames, columns=["Coefficients"])
  best_attributes["Coefficient Magnitudes"] = np.abs(best_attributes["Coefficients"])
  best_attributes = best_attributes["Coefficient Magnitudes"].sort_values(ascending=False)
  
  
  #Saving the coefficients and attributes
  coef_path = out_dir + "/best_coefficients.feather" 
  feather.write_dataframe(best_attributes.reset_index(), coef_path)
  
  
  #Scoring the model
  score = -np.sqrt(mean_squared_error(y_test, random_grid.best_estimator_.predict(X_test)))
  print(f"The Negative Root Mean Squared Error for the test score is {round(score,3)}")
  
  
  #Plot the residuals table
  sns.set_theme(style="whitegrid")
  X_test.index.name = "Test Data Observations"
  predict_test = random_grid.best_estimator_.predict(X_test)
  residual = y_test.copy()
  residual.name = "Residuals(hours)"
  residual_plot = sns.residplot(x=X_test.index, y=residual - predict_test,  color="g")
  residual_fig = residual_plot.get_figure()
  residual_dir = out_dir + "/residual_plot"
  residual_fig.savefig(residual_dir) 
  
  
  #Saving score in a pickle file  
  test_score_file = out_dir + "/test_score.pickle"
  pickle_out = open(test_score_file,"wb")
  pickle.dump(score, pickle_out)
  pickle_out.close()
  
  
if __name__ == "__main__":
  main(opt["--input_xtrain"], opt["--input_ytrain"], opt["--input_xtest"], opt["--input_ytest"], opt["--input_processor"], opt["--input_total_features"], opt["--out_dir"])
  
