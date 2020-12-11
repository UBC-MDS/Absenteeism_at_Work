# Absenteeism at work
# author: DSCI_522 Group 21
# date: 2020-12-02

all: results/correlation_matrix.png results/distribution_plot.png  results/frequency_plot.png results/non_RFE_CV_results.feather results/RFE_CV_results.feather results/best_coefficients.feather doc/absenteeism_predict_report.md

# download data
data/Absenteeism_at_work: script/download_data.R
	Rscript  script/download_data.R --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip"


# clean and save the data (e.g., scale and split into train & test)
results/train_df.feather results/test_df.feather : script/read_clean_split_data.py data/Absenteeism_at_work.csv
	python  script/read_clean_split_data.py --input="data/Absenteeism_at_work.csv" --out_dir="results"

# create exploratory data analysis 
results/correlation_matrix.png results/distribution_plot.png results/frequency_plot.png : script/eda.R results/train_df.feather
	Rscript script/eda.R --train="results/train_df.feather" --out_dir="results"

# pre-process the data
results/processor.pickle: script/preprocessing_machine_learning.py results/train_df.feather 
	python script/preprocessing_machine_learning.py --input_train="results/train_df.feather" --out_dir="results"

# fit the models and get test results from the best model
results/non_RFE_CV_results.feather results/RFE_CV_results.feather results/best_coefficients.feather results/test_score.pickle : script/machine_learning_model.py results/train_df.feather results/test_df.feather results/processor.pickle
	python script/machine_learning_model.py --input_train="results/train_df.feather" --input_test="results/test_df.feather" --input_processor="results/processor.pickle" --out_dir="results"

# render report
doc/absenteeism_predict_report.md: doc/absenteeism_predict_report.Rmd doc/absenteeism_refs.bib  
	Rscript -e "rmarkdown::render('doc/absenteeism_predict_report.Rmd', output_format = 'github_document')"

clean: 
	rm -rf data
	rm -rf results
	rm -rf doc/absenteeism_predict_report.md doc/absenteeism_predict_report.html