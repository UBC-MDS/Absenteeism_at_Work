Absenteeism Hours Predictor
===========================

-   Author: MDS DSCI 522 Group 21

About
-----

In this project we built three machine learning models, which are
`random forest regressor`,
`support vector machine regressor with linear kernel` and
`ridge regressor`, to make predictions on absenteeism time in hours from
“Absenteeism at work” dataset. There are 515 training and 222 test data
observations. We used 19 columns after dropping “ID” as our features,
and `Absenteeism time in hours` as the target. After extracting
categorical, binary, and ordinal features with preprocessing, we ended
up with 64 features. Our machine learning models did not perform a very
good job even after we applied recursive feature elimination. From
feature selection, we found that the most impactful feature is “Reason
for absence”. The best mean cross-validation *R*<sup>2</sup> score we
got is 0.09 from `ridge regressor`, and the other 2 validation scores
are 0.05 from `support vector machine regressor with linear kernel` and
-0.15 from `random forest regressor`. Thus, we used `ridge regression`
model on the test data and our test score is 0.085. Our unsatisfied
prediction results may affect the decision and judgement that an
employer make while dealing with absenteeism among employees. Thus, We
suggest that more sophisticated approaches on feature selection and
machine learning models should be applied on this data to improve the
prediction results; alternatively, a more representative and independent
dataset is needed to perform the prediction upon, in order to gain the
correct direction on absenteeism issues.

Our data set is chosen from the UCI Machine Learning Repository called
“Absenteeism at work Data Set”. The data set can be found
[here](https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work#) and
it is created by Andrea Martiniano, Ricardo Pinto Ferreira, and Renato
Jose Sassi from Postgraduate Program in Informatics and Knowledge
Management at Nove de Julho University, Rua Vergueiro(Andrea Martiniano,
Ricardo Pinto Ferreira, and Renato Jose Sassi 2010). The data was
collected at a courier company in Brazil and the database includes the
monthly records of absenteeism of 36 different workers over three years,
starting from July 2007, and how their changes affect their absence rate
over time. This data set contains 740 instances with 21 attributes,
including 8 categorical and 11 numerical features (excluding the target
`Absenteeism time in hours` and the drop feature `ID`). Each row
represents information about an employee with his/her situations of
absence, family, workload, and other factors that might be related to
his/her absence at work. Out of the considered attributes, the
absenteeism in hours is the target to predict with the provided
information, and the features are:

-   **ID** (feature that will be dropped): Individual Identification of
    each employee. There are 36 distinct individuals.

-   **Reason for absence** (categorical feature): Justification for the
    registered absence hours of each employee.

-   **Month of absence** (categorical feature): The month in which the
    absentee time is registered.

-   **Day of the week** (categorical feature): The five business days of
    the week.

-   **Seasons** (categorical feature): The four seasons of the year with
    summer (1), autumn (2), winter (3), spring (4).

-   **Transportation Expense** (numeric feature): Monthly transportation
    expense of each employee in dollars.

-   **Distance from residence to work** (numeric feature): Distance
    covered by each employee daily in kilometers.

-   **Service Time** (numeric feature): Service time of each employee in
    years.

-   **Age** (numeric feature): Age of each employee in years.

-   **Workload Average/day** (numeric feature): Workload of each
    employee per day.

-   **Hit target** (numeric feature): achievement percentage (%) of
    periodic goals for each employee.

-   **Disciplinary failure** (binary feature): Whether or not the
    employee received a disciplinary warning that month.

-   **Education** (ordinal feature): Level of education of each
    employee.

-   **Son** (numeric feature): Number of children of each employee.

-   **Social Drinker** (binary feature): Whether the employee is a
    social drinker or not.

-   **Social smoker** (binary feature): Whether the employee is a social
    smoker or not.

-   **Pet** (numeric feature): Number of pets of each employee.

-   **Weight** (numeric feature): Weight of each employee in kilograms.

-   **Height** (numeric feature): Height of each employee in
    centimeters.

-   **Body Mass Index** (numeric feature): Body mass percentage (%) of
    each employee.

Report
------

The final report can be found
[here](http://htmlpreview.github.io/?https://raw.githubusercontent.com/UBC-MDS/dsci-522_group-21/main/doc/absenteeism_predict_report.html)

Usage
-----

The replication of this project can be done by installing the following
dependencies and running the following command from the root directory
of this project to unzip the data:

    # Using R to download data:
    Rscript  script/download_data.R --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip"

    # or, alternatively using Python to download data:
    python  script/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip" --out_path="data"

    # load, clean and save the data
    python  script/read_clean_split_data.py --input="data/Absenteeism_at_work.csv" --out_dir="results"

    # create exploratory data analysis
    Rscript script/eda.R --train="results/train_df.feather" --out_dir="results"

    # pre-process the data
    python script/preprocessing_machine_learning.py --input_train="results/train_df.feather" --out_dir="results"

    # fit the models and get test results from the best model
    python script/machine_learning_model.py --input_train="results/train_df.feather" --input_test="results/test_df.feather" --input_processor="results/processor.pickle" --out_dir="results"

    # render the report
    Rscript -e "rmarkdown::render('doc/absenteeism_predict_report.Rmd', output_format = 'github_document')"

Dependencies
------------

-   Python 3.7.3 and Python packages:

    -   docopt==0.6.2

    -   pandas==0.24.2

    -   feather-format==0.4.0

    -   scikit-learn&gt;=0.23.2

    -   requests==2.22.0

-   R version 3.6.1 and R packages:

    -   tidyverse==1.2.1

    -   knitr==1.26

    -   feather==0.3.5

    -   dplyr==1.0.2

    -   ggcorrplot==0.1.3

    -   ggthemes==4.2.0

References
==========

Andrea Martiniano, Ricardo Pinto Ferreira, and Renato Jose Sassi. 2010.
“UCI: Machine Learning Repository.” Universidade Nove de Julho -
Postgraduate Program in Informatics; Knowledge Management.
<https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work#>.
