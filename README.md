In this project, we will try to address the following predictive
question:

> Based on some given information of an employee, regarding the
> personal, working and health ambits, how many hours of absence would
> be expected from that worker in a specific month?

### Background Information

Absenteeism in the workplace is the habitual absence behavior from work
without a valid reason(*2016 Absence Management Annual Survey Report*
2016). It is a very common case experienced by employers and it has
become a serious problem that employers always want to deal with. The UK
Chartered Institute of Personnel and Development(also known as CIPD)
estimated that employers had to pay £595 on each employee per year and
which is caused by 7.6 absent days from each worker on average in
2013(Cucchiella, Gastaldi, and Ranieri 2014). In addition to higher
financial costs, absenteeism might lead to reduced productivity levels
and low morale in workplaces, which affects the overall operation of an
organization.

### Problem Description

A statistical way to deal with this problem is that we can figure out
the most impactful factors that cause absenteeism and make predictions
according to the characteristics of different individuals. If employers
can use the results to predict absenteeism among employees, they can
make effective plans in advance to deal with the upcoming problems and
reduce extra costs caused by absenteeism.

To help employers deal with absenteeism related issues, we would like to
use a statistical method to make predictions and find the most
influential factors on absenteeism. Our project focuses on using machine
learning methods to learn from absenteeism at work data set and make
predictions on different individuals after finding out the main factors
that would lead to absence from work.

### Data Set Description

We chose a data set from the UCI Machine Learning Repository called
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

<!-- -->

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

### Potential Approaches

The predictive project will consist of three phases.

1.  Before building the model, we will split the data into a training
    and a test data set (70% : 30% split). Expecting that the test data
    set is representative of the deployment data that the model is going
    to face in the future. The test data set will be saved and only used
    to score the trained predictive model.

2.  An exploratory data analysis (EDA) will be performed in the training
    data set only to assess whether there are considerable correlations
    between features affecting the considerations for the model
    selection, as well as whether there are correlations between
    features and the target, to obtain a general idea of the most
    influential features associated with the change in target. We will
    also take into account of the distribution of each feature and the
    target to detect any outliers or class imbalances. Therefore, we
    will generate a correlation plot and some distribution plots for
    each feature. In particular, we will also generate a frequency table
    and a histogram for the `Reason of Absence` feature to understand
    both which justifications are more common, and which of them are
    more severe in affecting the absenteeism hours.

3.  Post EDA, we will use supervised machine learning models to perform
    prediction and obtain the most suitable algorithm for our Abseentism
    hours prediction task. The models we have chosen including
    `dummy regressor`, to use as a baseline model, `ridge regression`,
    `decision tree regression` (mostly for its speed and easy model
    interpretation), `random forest regression`, and
    `support vector machine with RBF kernel`. Support vector machine was
    chosen mostly for its accuracy when a considerable amount of
    features are utilized. Moreover, ridge regression was selected to
    deal with multicollinearity between the features, as it is expected
    that features like body mass index, height, and weight have an
    existing positive correlation. For each model, hyperparameter
    optimization will be conducted to determine the most representative
    figures. Is it important to inform the optimization bias during
    cross\_validation will be taken into consideration. For evaluation
    metrics, `mean squared error` and `R-square` are going to be used to
    assess how these models perform. We will evaluate the overall
    performance for each model and the result will be included in the
    final report.

### Usage

The replication of this project can be done by installing the following
dependencies and running the following command from the root directory
of this project to unzip the data:

Using `R`:

`Rscript  script/download_data.R --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip"`

or, alternatively using `Python`:

`python  script/download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip" --out_path="data"`

The data file will be saved in a folder called `data`.

### Dependencies

-   ipykernel

-   matplotlib&gt;=3.2.2

-   scikit-learn&gt;=0.23.2

-   pandas&gt;=1.1.3

-   requests&gt;=2.24.0

-   graphviz

-   python-graphviz

-   altair&gt;=4.1.0

-   jinja2

-   pip&gt;=20

-   pandas-profiling&gt;=1.4.3

-   pip:

    -   psutil&gt;=5.7.2

    -   xgboost&gt;=1.\*

    -   lightgbm&gt;=3.\*

### References

*2016 Absence Management Annual Survey Report*. 2016. Chartered
Institute of Personnel; Development.
<https://www.cipd.co.uk/Images/absence-management_2016_tcm18-16360.pdf>.

Andrea Martiniano, Ricardo Pinto Ferreira, and Renato Jose Sassi. 2010.
“UCI: Machine Learning Repository.” Universidade Nove de Julho -
Postgraduate Program in Informatics; Knowledge Management.
<https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work#>.

Cucchiella, Federica, Massimo Gastaldi, and Luigi Ranieri. 2014.
“Managing Absenteeism in the Workplace: The Case of an Italian
Multiutility Company.” *Procedia - Social and Behavioral Sciences* 150:
1157–66. <https://doi.org/https://doi.org/10.1016/j.sbspro.2014.09.131>.
