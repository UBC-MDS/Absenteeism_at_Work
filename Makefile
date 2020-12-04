# Absenteeism data pipe
# author: Yiki Su
# date: 2020-12-01

all: doc/absenteeism_predict_report.html doc/absenteeism_predict_report.md

# download data
data/Absenteeism_at_work.csv: script/download_data.py
	python script/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip" --out_path="data"

# pre-process data (e.g., split the data into train & test)
results/train_df.feather results/test_df.feather: data/Absenteeism_at_work.csv
	python script/read_clean_split_data.py --input="data/Absenteeism_at_work.csv" --out_dir="results" 

# exploratory data analysis - 
results/correlation_matrix.png results/distribution_plot.png results/frequency_plot.png: results/train_df.feather
	Rscript script/eda.R --train="results/train_df.feather" --out_dir="results"

# pre-process the data
results/processor.pickle: results/train_df.feather
	python script/preprocessing_machine_learning.py --input_train="results/train_df.feather" --out_dir="results"

# fit the models and get test results from the best model
results/best_coefficients.feather results/non_RFE_CV_results.feather results/RFE_CV_results.feather: results/train_df.feather results/test_df.feather results/processor.pickle
	python script/machine_learning_model.py --input_train="results/train_df.feather" --input_test="results/test_df.feather" --input_processor="results/processor.pickle" --out_dir="results"

# render the report
doc/absenteeism_predict_report.html doc/absenteeism_predict_report.md : doc/absenteeism_predict_report.Rmd results/best_coefficients.feather results/correlation_matrix.png results/distribution_plot.png results/frequency_plot.png results/non_RFE_CV_results.feather results/RFE_CV_results.feather results/test_score.pickle
	Rscript -e "rmarkdown::render('doc/absenteeism_predict_report.Rmd', output_format = 'github_document')"
	
clean: 
	rm -rf data
	rm -rf results
	rm -rf doc/absenteeism_predict_report.html doc/absenteeism_predict_report.md
			