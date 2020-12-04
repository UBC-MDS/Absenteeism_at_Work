Predicting absenteeism hours at work from different features
================
MDS DSCI 522 Group 21
27/11/2020

Summary
=======

> In this project, we are trying to address the following **predictive
> question**:

> Based on some given information of an employee, regarding personal,
> working and health ambits, how many hours of absence would be expected
> from that employee?

In this project, we built three machine learning regressor models:
`random forest regressor`,
`support vector machine regressor with linear kernel` and
`ridge regressor` to make predictions on absenteeism time in hours from
the “Absenteeism at work” dataset.

Our final model `support vector machine regressor with linear kernel`
performed a decent job on an unseen test data set, with `negative RMSE`
score of -5.825. On 222 test data cases, the average hours that our
model missed to predict is 5.825 hours, which is not bad at all.
However, in both the train and test dataset, our predictor tends to over
predict when the actual absenteeism hours are low and under predict in
the case of actual absenteeism hours are high.

Since our prediction results may affect the decision and judgement that
an employer makes when dealing with absenteeism among employees, we
suggest that more sophisticated approaches on machine learning algorithm
and feature selection should be conducted to improve the prediction
model before it is being used to direct on absenteeism issues at the
workplace.

Introduction
============

Absenteeism in the workplace is the habitual absence behavior from work
without a valid reason(“2016 Absence Management Annual Survey Report”
2016). It is a very common case experienced by employers and it has
become a serious problem that employers always want to deal with. The UK
Chartered Institute of Personnel and Development(also known as CIPD)
estimated that employers had to pay £595 on each employee per year and
which is caused by 7.6 absent days from each worker on average in
2013(Cucchiella, Gastaldi, and Ranieri 2014). In addition to higher
financial costs, absenteeism might lead to reduced productivity levels
and low morale in workplaces, which affects the overall operation of an
organization.

Here we would like to experiment if we could use a machine learning
model to make predictions and find the most influential features on
absenteeism. If employers can use the results to predict absenteeism
among employees, they can make effective plans in advance to deal with
the upcoming problems and reduce extra costs caused by absenteeism.

Data
====

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
including 6 categorical and 9 numerical features (excluding the target
`Absenteeism time in hours` and the drop feature `ID`,
`Disciplinary failure`, `Body mass index`, `Service time`, and
`Month of absence`). Each row represents information about an employee’s
situations of absence, family, workload, and other factors that might be
related to absence at work. Out of the considered attributes, the
absenteeism in hours is our prediction target.

Methods
=======

### Analysis tools

We used both R(R Core Team 2019) and Python(Van Rossum and Drake 2009)
programming languages to perform this prediction task. The following R
and Python packages were used: tidyverse(Wickham 2017), dplyr(Wickham et
al. 2020), knitr(Xie 2014), ggcorrplot(Kassambara 2019), ggthemes(Arnold
2019), docopt R(de Jonge 2018), docopt Python(Keleshev 2014), feather
Python(McKinney 2019), os(Van Rossum and Drake 2009), Pandas(McKinney
2010), and scikit-learn(Buitinck et al. 2013). The code used to perform
the analysis and create this report can be found
[here](https://github.com/UBC-MDS/dsci-522_group-21).

### Preliminary data Analysis

Prior to preliminary data analysis and building the model, we splitted
the data into a 515 training set and a 222 test set (70% : 30% split).
We assume that the test data set is representative of the deployment
data that the model is going to face in the future. Therefore the test
data set are saved and only be used to predict and score the trained
model. Then we performed some exploratory data analysis (EDA) on the
training data only:

From figure 1, We observed that there are some considerable correlations
between features, as well as there is some correlations between the
target and respective features. For example, `Disciplinary failure` and
`Reason for absence`; `Hit target` and `Month of absence`;
`body mass index` and `weight`; `weight` and `service time` seem to be
highly correlated features. As a result, we decided to drop the
`Disciplinary failure`, `Body mass index`, `Service time`, and
`Month of absence` features to better deal with multicollinearity
issues.

<div class="figure">

<img src="../results/correlation_matrix.png" alt="Figure 1. Correlation matrix between all features and the target" width="100%" />
<p class="caption">
Figure 1. Correlation matrix between all features and the target
</p>

</div>

We looked into the distributions (figure 2) of each feature and the
target and we detect many outliers in the target. Therefore, in both of
our train and test data, we removed some extreme outliers.

<div class="figure">

<img src="../results/distribution_plot.png" alt="Figure 2. Frequency distributions for all features and the target" width="100%" />
<p class="caption">
Figure 2. Frequency distributions for all features and the target
</p>

</div>

We examined the distribution for the particular feature
`Reason of Absence` (figure 3), which has one of the relatively highest
correlation with the target, and observe that justifications 22 (medical
consultation) and 27 (Dental Consultation) are the most common, causing
the reasons for absence in 191 out of the 508 observations taken.

<div class="figure">

<img src="../results/frequency_plot.png" alt="Figure 3. Reasons of Absence feature distribution" width="100%" />
<p class="caption">
Figure 3. Reasons of Absence feature distribution
</p>

</div>

### Data preprocessing and transforming

We built a preprocessing and transforming pipeline for all features:
simple scaler for numeric features, one hot encoding for both
categorical and binary features, ordinal encoding for ordinal features,
and we dropped features `ID`, `Disciplinary failure`, `Body mass index`,
`Service time`, and `Month of absence` to better deal with
multicollinearity issues. In addition, in both of our train and test
data, we removed some extreme outliers to better deal with extreme edge
cases.

### Prediction models & evaluation metric

Post EDA, we are ready to use supervised machine learning models to
perform prediction and to obtain the most suitable algorithm for our
Abseentism prediction task. The models we chose are:

-   `support vector machine with linear kernel` - we chose this model
    for its accuracy when a considerable amount of features are
    utilized.

-   `ridge regressor` - we selected this model to better deal with
    multicollinearity between the features.

-   `random forest regressor` - we chose this model for its efficiency
    and easiness to view relative feature importance.

For evaluation metric, both *R*<sup>2</sup> score and
`negative root mean squared error` (`negative RMSE`) are used to assess
how these models perform. Specifically, *R*<sup>2</sup> measures how
well the models adapt and represent the training data, with 1 being
making a perfect prediction and 0 being not having any predicting power;
whereas `negative RMSE` measures how many absenteeism hours our
prediction model misses in the validation / test data set. More
importantly, we will focus on the `negative RMSE` because this
measurement matters to our prediction task the most.

Prediction results
==================

### Cross validation

First, We performed cross validation on the train data set with 5
cross-validation folds using all 3 machine learning models. Table 1
shows the default mean cross-validation (cv) and train *R*<sup>2</sup>
and `negative RMSE` scores for each machine learning model. The key
takeaway from this table is that the
`support vector machine with linear kernel` model seems to be a good
candidate predictor model with least overfitting issues and similar
`negative RMSE` mean cv scores of around -5 to its peer models.

Given these closely performing `negative RMSE` mean cv scores across all
three models, we proceed with feature selection to try to filter down
the most suitable model to use along with its associated most important
features.

| index        | Linear SVM |     Ridge | Random Forest |
|:-------------|-----------:|----------:|--------------:|
| fit\_time    |  0.2268667 | 0.2044175 |     2.7218767 |
| score\_time  |  0.0674204 | 0.0959224 |     0.0918088 |
| test\_score  |  0.0631696 | 0.0569267 |    -0.1527934 |
| train\_score |  0.0704791 | 0.2766440 |     0.8357503 |

Table 1. Non feature selection mean cross validation R-square scores of
all three machine learning models

### Feature selection & hyperparameter tuning

We used recursive feature elimination and cross-validated selection
(`RFECV`) on the 3 machine learning models and we performed
cross-validation (cv) again. Table 2 shows the mean cv and train
*R*<sup>2</sup> and `negative RMSE` scores based on the most important
features selected associated with each of the 3 models. The key takeaway
from this table is that the `support vector machine with linear kernel`
model seems to be the best predictor model with least overfitting issues
and this time a better `negative RMSE` mean cv scores of -4.81 than its
peer models.

| index        | Linear SVM |     Ridge | Random Forest |
|:-------------|-----------:|----------:|--------------:|
| fit\_time    |  7.5928011 | 8.2742509 |     8.0435517 |
| score\_time  |  0.0466359 | 0.0368004 |     0.0778455 |
| test\_score  |  0.0475836 | 0.0937344 |    -0.1460047 |
| train\_score |  0.0505426 | 0.1724282 |     0.2736223 |

Table 2. Feature selection mean cross validation R-square scores of all
three machine learning models

As a result, we picked `support vector machine with linear kernel` as
our final prediction model. The 19 most important features (out of 49
total transformed features) of our final prediction model selected by
REFCV are listed in descending order shown in table 3. with the majority
of important features coming from `Reason for absence`. Furthermore
`Tuesday` and whether or not an employee is a `social drinker` being
another 2 important features out of the 19.

In addition, we performed hyperparameter tuning on our final prediction
model using `random search cross validation` and the best
hyperparameters given are gamma of 0.1 and C of 1, while hyperparameter
tuning did not improve further our -4.81 `negative RMSE` mean cv scores.

| index                  | Coefficients |
|:-----------------------|-------------:|
| Reason for absence\_0  |    -4.076700 |
| Reason for absence\_2  |     1.329469 |
| Reason for absence\_6  |     5.686487 |
| Reason for absence\_9  |     8.113978 |
| Reason for absence\_12 |     6.904067 |
| Reason for absence\_13 |     6.304106 |
| Reason for absence\_19 |     9.179391 |
| Reason for absence\_23 |    -6.295894 |
| Reason for absence\_25 |    -4.157175 |
| Reason for absence\_27 |    -5.816594 |
| Reason for absence\_28 |    -5.833860 |
| Month of absence\_7    |     3.880589 |
| Disciplinary failure   |    -4.076700 |

Table 3. Most important Features selected with associated coefficients
under ridge regressor prediction model

### Test result

Now we are ready to use our final prediction model
`support vector machine with linear kernel` on our test data set. The
final test `negative RMSE` score is -5.825, which is very close to the
cross validation scores we got previously, which is a good indicator
that our model generalizes well on the unseen test set.

Lastly, we included the residual plot in Figure 4, which shows the
residuals of our predictions on Y axis and all the actual test targets
on the X axis. We can see that the majority of our prediction residuals
are clustered around 0 throughout the entire test data set, and our
prediction model is performing a decent job in predicting the hours of
absence from a worker with some errors.

Discussions
===========

### Critique

There are limitations and assumptions associated with our prediction
task:

-   The dataset is collected from one single courier company in Brazil,
    which means that the data might not be independent and
    representative of the population that we are interested in
    predicting.

-   From the preliminary data analysis, we see that there is no strong
    correlation between each single feature and the target, which is a
    signal that there might not be a great representation of target from
    the given features. There are obvious multicollinearity in between
    features, which we decided on removing prior to training our machine
    learning models, and this might not have been the best approach to
    deal with multicollinearity.

-   In addition, from the frequency distributions, there are many
    outliers in our target, so we decided on removing some prior to
    training our models, which also might not have been the most
    effective way to deal with outliers, and could potentially make our
    prediction model more sensitive and less robust when it comes to
    predicting on extreme cases.

### Future directions

Given the current Machine Learning tools we have learned so far, we were
able to answer our predictive question in a basic manner. If we were to
have more time to explore deeper, we would 1. research into more
advanced machine learning models that particularly deal with
multicollinear data and outlier data; 2. find and use a more
representative and independent dataset that could better represent the
population to perform analysis and prediction on.

References
==========

<div id="refs" class="references hanging-indent">

<div id="ref-CIPD">

“2016 Absence Management Annual Survey Report.” 2016. Chartered
Institute of Personnel; Development.
<https://www.cipd.co.uk/Images/absence-management_2016_tcm18-16360.pdf>.

</div>

<div id="ref-data">

Andrea Martiniano, Ricardo Pinto Ferreira, and Renato Jose Sassi. 2010.
“UCI: Machine Learning Repository.” Universidade Nove de Julho -
Postgraduate Program in Informatics; Knowledge Management.
<https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work#>.

</div>

<div id="ref-ggthemes">

Arnold, Jeffrey B. 2019. *Ggthemes: Extra Themes, Scales and Geoms for
’Ggplot2’*. <https://CRAN.R-project.org/package=ggthemes>.

</div>

<div id="ref-sklearn_api">

Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa,
Andreas Mueller, Olivier Grisel, Vlad Niculae, et al. 2013. “API Design
for Machine Learning Software: Experiences from the Scikit-Learn
Project.” In *ECML Pkdd Workshop: Languages for Data Mining and Machine
Learning*, 108–22.

</div>

<div id="ref-book2">

Cucchiella, Federica, Massimo Gastaldi, and Luigi Ranieri. 2014.
“Managing Absenteeism in the Workplace: The Case of an Italian
Multiutility Company.” *Procedia - Social and Behavioral Sciences* 150:
1157–66. <https://doi.org/https://doi.org/10.1016/j.sbspro.2014.09.131>.

</div>

<div id="ref-docopt">

de Jonge, Edwin. 2018. *Docopt: Command-Line Interface Specification
Language*. <https://CRAN.R-project.org/package=docopt>.

</div>

<div id="ref-ggcorrplot">

Kassambara, Alboukadel. 2019. *Ggcorrplot: Visualization of a
Correlation Matrix Using ’Ggplot2’*.
<https://CRAN.R-project.org/package=ggcorrplot>.

</div>

<div id="ref-docoptpython">

Keleshev, Vladimir. 2014. *Docopt: Command-Line Interface Description
Language*. <https://github.com/docopt/docopt>.

</div>

<div id="ref-featherpy">

McKinney, Wes. 2019. *Feather: Simple Wrapper Library to the Apache
Arrow-Based Feather File Format*. <https://github.com/wesm/feather>.

</div>

<div id="ref-mckinney-proc-scipy-2010">

McKinney. 2010. “Data Structures for Statistical Computing in Python.”
In *Proceedings of the 9th Python in Science Conference*, edited by
Stéfan van der Walt and Jarrod Millman, 56–61.
[https://doi.org/ 10.25080/Majora-92bf1922-00a](https://doi.org/%2010.25080/Majora-92bf1922-00a%20).

</div>

<div id="ref-R">

R Core Team. 2019. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-Python">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-tidyverse">

Wickham, Hadley. 2017. *Tidyverse: Easily Install and Load the
’Tidyverse’*. <https://CRAN.R-project.org/package=tidyverse>.

</div>

<div id="ref-dplyr">

Wickham, Hadley, Romain Fran�ois, Lionel Henry, and Kirill M�ller. 2020.
*Dplyr: A Grammar of Data Manipulation*.
<https://CRAN.R-project.org/package=dplyr>.

</div>

<div id="ref-knitr">

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research
in R.” In *Implementing Reproducible Computational Research*, edited by
Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman;
Hall/CRC. <http://www.crcpress.com/product/isbn/9781466561595>.

</div>

</div>
