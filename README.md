# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
 The project will help us to identify the customers who are likely to churn

## Files and data description
The project contains two important files:
   1. churn_library.py                   : This file contain code for loading data, performing feature engineering, training the model and saving it. 
   2. churn_script_logging_and_tests.py  : This file contains the unit test for testing the functions of churn_library.py
<br />
   
.
 * [data](./data)
   * [bank_data.csv](./data/bank_data.csv)
 * [requirements_py3.6.txt](./requirements_py3.6.txt)
 * [requirements_py3.8.txt](./requirements_py3.8.txt)
 * [Guide.ipynb](./Guide.ipynb)
 * [churn_notebook.ipynb](./churn_notebook.ipynb)
 * [images](./images)
   * [eda](./images/eda)
     * [Distribution_of_Churn.png](./images/eda/Distribution_of_Churn.png)
     * [Distribution_of_Customer_Age.png](./images/eda/Distribution_of_Customer_Age.png)
     * [Distribution_of_Marital_Status.png](./images/eda/Distribution_of_Marital_Status.png)
     * [Distribution_of_Total_Trans_Ct.png](./images/eda/Distribution_of_Total_Trans_Ct.png)
     * [Heatmap_distribution.png](./images/eda/Heatmap_distribution.png)
   * [results](./images/results)
   * [RF_Metrices_result.png](./images/results/RF_Metrices_result.png)
   * [LR_Metrices_result.png](./images/results/LR_Metrices_result.png)
   * [ROC_plot.png](./images/results/ROC_plot.png)
   * [Feature_Importance.png](./images/results/Feature_Importance.png)
 * [churn_script_logging_and_tests.py](./churn_script_logging_and_tests.py)
 * [logs](./logs)
   * [churn_library.log](./logs/churn_library.log)
 * [churn_library.py](./churn_library.py)
 * [README.md](./README.md)
 * [__pycache__](./__pycache__)
   * [churn_script_logging_and_tests.cpython-38-pytest-7.1.2.pyc](./__pycache__/churn_script_logging_and_tests.cpython-38-pytest-7.1.2.pyc)
   * [churn_library.cpython-38.pyc](./__pycache__/churn_library.cpython-38.pyc)
 * [models](./models)
   * [logistic_model.pkl](./models/logistic_model.pkl)
   * [rfc_model.pkl](./models/rfc_model.pkl)
 * [tree-md](./tree-md)(mlops_proj1) 

The data has been downloaded from Kaggle. 
  

## Running Files
  1. All the python package dependencies need to be installed. These are present in the file requirements_py3.8.txt. Use the command:
      <br />
      pip install -r requirements_py3.8.txt
  2. To run the file use the following commands:
      <br />
       ipython churn_library.py
       <br />
       ipython churn_script_logging_and_tests.py
 



