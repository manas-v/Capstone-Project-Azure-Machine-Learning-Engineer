# Capstone-Project-Azure-Machine-Learning-Engineer
This is the final project of the Udacity Azure ML Nanodegree. In this project, we will create two models on the 'IBM HR Analytics Employee Attrition & Performance' dataset: One using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.

### Project Workflow
![image](https://user-images.githubusercontent.com/59551550/107699098-9518cc00-6cdb-11eb-8703-e9ccaf1a7b82.png)

## Dataset

### Overview
Attrition has always been a major concern in any organization. The IBM HR Attrition Case Study is a fictional dataset that aims to identify important factors that might be influential in determining which employee might leave the firm and who may not.

#### Dataset Attributes
1. Age
2. Attrition
3. BusinessTravel
4. DailyRate
5. Department
6. DistanceFromHome
7. Education
8. EducationField
9. EmployeeCount
10. EmployeeNumber
11. EnvironmentSatisfaction
12. Gender
13. HourlyRate
14. JobInvolvement
15. JobLevel
16. JobRole
17. JobSatisfaction
18. MaritalStatus
19. MonthlyIncome
20. MonthlyRate
21. NumCompaniesWorked
22. Over18
23. OverTime
24. PercentSalaryHike
25. PerformanceRating
26. RelationshipSatisfaction
27. StandardHours
28. StockOptionLevel
29. TotalWorkingYears
30. TrainingTimesLastYear
31. WorkLifeBalance
32. YearsAtCompany
33. YearsInCurrentRole
34. YearsSinceLastPromotion
35. YearsWithCurrManager

### Task
The Dataset consists of 35 columns, through which we aim to predict whether an employee will leave the job or not. This is a binary classification problem, where the outcome 'Attrition' will either be 'true' or 'false'. In this experiment, we will be using HyperDrive and AutoML to find the best prediction for the given Dataset. We will then deploy the model with the best prediction and interact with the deployment.

### Access
The dataset is available on Kaggle as 'IBM HR Analytics Employee Attrition & Performance' dataset, but for this project, the dataset has been uploaded onto GitHub and is accessed through the following URI: 'https://raw.githubusercontent.com/manas-v/Capstone-Project-Azure-Machine-Learning-Engineer/main/WA_Fn-UseC_-HR-Employee-Attrition.csv'
We then use Tabular Dataset Factory's ```Dataset.Tabular.from_delimited_files()``` to get the data from the url and save it to the datastore by using ```dataset.register()```

## Automated ML
AutoML or Automated ML is the process of automating the task of machine learning model development. Using this feature, you can predict the best ML model, and its hyperparameters suited for your problem statement.

This is a binary classification problem with the label column 'Attrition' having output as 'true' or 'false'. The experiment timeout is 20 mins, a maximum of 5 concurrent iterations take place together, the primary metric for the run is AUC_weighted.
The AutoML configurations used for this experiment are:

| Configuration | Value | Explanation |
|    :---:     |     :---:      |     :---:     |
| experiment_timeout_minutes | 20 | Maximum amount of time in minutes that all iterations combined can take before the experiment terminates |
| max_concurrent_iterations | 5 | Represents the maximum number of iterations that would be executed in parallel |
| primary_metric | AUC_weighted | The metric that Automated Machine Learning will optimize for model selection |
| compute_target | cpu_cluster(created) | The Azure Machine Learning compute target to run the Automated Machine Learning experiment on |
| task | classification | The type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem to solve |
| training_data | dataset(imported) | The training data to be used within the experiment |
| label_column_name | Attrition | The name of the label column |
| path | ./capstone-project | The full path to the Azure Machine Learning project folder |
| enable_early_stopping | True | Whether to enable early termination if the score is not improving in the short term |
| featurization | auto | Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used |
| debug_log | automl_errors.log | The log file to write debug information to |

### AutoML Results
After running the AutoML pipeline, the best performing model is found to be VotingEnsemble with an AUC_weighted value of 0.83328615. VotingEnsemble combines conceptually different machine learning classifiers and uses a majority vote or the average predicted probabilities (soft vote) to predict the class labels. This is method balances out the individual weaknesses of the considered classifiers.

The AutoML Voting Classifier for this run is made up of a combination of 11 classifiers with different hyperparameter values and normalization/scaling techinques. The 11 estimators used in the run weigh 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.23076923076923078, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693 respectively.

Considering the randomforestclassifier, the model with one of the highest weights i.e. 0.23076923076923078
The hyperparameters generated for the model are: 
```
14 - maxabsscaler
{'copy': True}

14 - randomforestclassifier
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'log2',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.01,
 'min_samples_split': 0.01,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 10,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}
```
### AutoML Screenshots

Output of RunDetails() widget
![image](https://user-images.githubusercontent.com/59551550/107697334-1ae74800-6cd9-11eb-8f5c-30af874ded1c.png)

Visualization of results
![image](https://user-images.githubusercontent.com/59551550/107697353-233f8300-6cd9-11eb-8f01-98e8496e0708.png)

Best performing model
![image](https://user-images.githubusercontent.com/59551550/107697417-34888f80-6cd9-11eb-91c7-d7248761c198.png)
![image](https://user-images.githubusercontent.com/59551550/107697497-4ff39a80-6cd9-11eb-84a5-d9e87707bdb7.png)

### Future work AutoML
Cross-Validation - Change the number of cross-validation folds in the AutoML run.
Primary metric - Attempting to look at other primary metrics too, incase they are more suitable for the model.
AutoML configurations - Use different AutoML configurations like experiment timeout, max concurrent iterations, etc, and observe the change in result.

## Hyperparameter Tuning using HyperDrive

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
The problem statement being a binary classification problem, the model used in it is the DecisionTreeClassifier. Decision Trees are simple to understand and to interpret, they are easy to visualize, require little data preparation. 

The hyperdrive configuration used for this experiment are as follows:

In this experiment, the early stopping policy used is Bandit Policy. Bandit policy is based on the difference in performance from the current best run, called 'slack'. Here the runs terminate where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.
```
early_termination_policy = BanditPolicy(slack_factor=0.1,evaluation_interval=3)
```

In this experiment, the parameter sampler used is Random Sampling. In random sampling, hyperparameter values are randomly selected from among the given search space. Random sampling is chosen because it supports discrete and continuous hyperparameters, and early termination of low-performance runs.
```
param_sampling = RandomParameterSampling({"--criterion": choice("gini", "entropy"),"--splitter": choice("best", "random"), "--max_depth": choice(3,4,5,6,7,8,9,10)})
```

In this experiment, the estimator used is SKLearn. An estimator that will be called with sampled hyper parameters.
```
estimator = SKLearn(source_directory=".", compute_target=cpu_cluster, entry_script="train.py")
```

| Configuration | Value | Explanation |
|    :---:     |     :---:      |     :---:     |
| hyperparameter_sampling | Value | Explanation |
| policy | early_termination_policy | The early termination policy to use |
| primary_metric_name | AUC_weighted | The name of the primary metric reported by the experiment runs |
| primary_metric_goal | PrimaryMetricGoal.MAXIMIZE | One of maximize / minimize. It determines if the primary metric has to be minimized/maximized in the experiment runs' evaluation |
| max_total_runs | 12 | Maximum number of runs. This is the upper bound |
| max_concurrent_runs | 4 | Maximum number of runs to run concurrently. |
| estimator | 4 | An estimator that will be called with sampled hyper parameters |


The Hyperparameters for the Decision Tree are:
| Hyperparameter | Value | Explanation |
|    :---:     |     :---:      |     :---:      |
| criterion | choice("gini", "entropy") | The function to measure the quality of a split. |
| splitter | choice("best", "random") | The strategy used to choose the split at each node. |
| max_depth | choice(3,4,5,6,7,8,9,10) | The maximum depth of the tree. |

### HyperDrive Results
The best result using HyperDrive was Decision Tree with Parameter Values as criterion = gini, max_depth = 4, splitter = best. The AUC_weighted of the Best Run is 0.7214713617767388.

### HyperDrive Screenshots

Output of RunDetails() widget
![image](https://user-images.githubusercontent.com/59551550/107696551-1e2e0400-6cd8-11eb-993f-b7368bee261e.png)

Visualization of results
![image](https://user-images.githubusercontent.com/59551550/107696666-43227700-6cd8-11eb-82c3-2864a49fcf96.png)
![image](https://user-images.githubusercontent.com/59551550/107696758-5d5c5500-6cd8-11eb-96bb-29c0ef459106.png)

Best performing model
![image](https://user-images.githubusercontent.com/59551550/107697040-b7f5b100-6cd8-11eb-8aa8-5fd802d54e5f.png)
![image](https://user-images.githubusercontent.com/59551550/107697155-dc518d80-6cd8-11eb-9cbd-e402988a9e6d.png)

### Future work HyperDrive
Model selection - Select a different classification ML algorithm to apply.
Sampling - Other parameter sampling methods to use over the hyperparameter space could be implemented i.e. Grid sampling, Bayesian sampling.
Early termination - Use other early termination policies such as Median stopping policy, Truncation selection policy. Different Early termination policies could be applied to keep the run most time/cost-efficient, yet having the best results.
Resource allocation - Different resource allocation in terms of max_total_runs, max_duration_minutes or max_concurrent_runs for HyperDrive configuration. 

## Model Deployment
Among AutoML and HyperDrive, the AUC_weighted of both were 0.83328615 and 0.7214713617767388. The AutoML being the one with the better results, we will deploy it.

The workflow for deploying a model to Azure CLI is
1. Register the model - A registered model is a logical container for one or more files that make up your model. Here we use the registered AutoML model.
2. Prepare an inference configuration (unless using no-code deployment) - An inference configuration describes how to set up the web-service containing your model.
3. Prepare an entry script (unless using no-code deployment) - The entry script receives data submitted to a deployed web service and passes it to the model. It then takes the response returned by the model and returns that to the client.
4. Choose a compute target - The compute target you use to host your model will affect the cost and availability of your deployed endpoint.
5. Deploy the model to the compute target - Before deploying your model, you must define the deployment configuration. The deployment configuration is specific to the compute target that will host the web service. The model will now deploy.
6. Test the resulting web service - You can test the model by querying the endpoint and sending sample input data to get JSON response.

Deployment Healthy
![image](https://user-images.githubusercontent.com/59551550/107698322-62ba9f00-6cda-11eb-861e-9015135d394e.png)

## Standout Suggestions
### Enable logging
Application Insights is used to detect anomalies, visualize performance, etc. It can be enabled before or after a deployment is created. For this experiment, we enable logging for the deployed model by running the logs.py script.

Logging enabled for endpoint
![image](https://user-images.githubusercontent.com/59551550/107698330-677f5300-6cda-11eb-999d-49e869699264.png)

## Screen Recording
Link to Screen Recording: https://youtu.be/zp1xjkhsK9k
