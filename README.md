# Project 2

Authors: Ali Nazim, Christopher Kuzemka

# General Requirements

- A Google Cluster with a Spark HDFS driver set up
- Python 3.11+


## Kaggle API Installation for Cluster

To access some of data in this project, you will need a Kaggle API key embedded into your cluster. [Please follow the instructions from Kaggle's documentation found by clicking here](https://www.kaggle.com/docs/api). You will need to download an API key locally, upload to main username of your cloud, and then move from your username to the root user of your cloud pending password authentication. This allows specific shell scripts to properly access some Kaggle datasets. 

# Project Summary

This project features a modularized process for running classification models in a Spark environment. Processes displayed here involve the use of PySpark's MLlib to process multiple Logistic Regression models for 4 unique problems. No ML Pipeline was instantiated due to little scope of the project in terms of cluster hardware and small data. 

**Part 1**

Part 1 showcases a Logistic Regression model to find probabilities of "toxic-like" comments within a corpus The data resides on Kaggle and [can be found here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). This dataset is best downloaded locally as it is relatively small and easy to manipulate. To get on cloud, you must move it locally via terminal to the cloud. More details inside sub-directory `src/part_1`

The Logistic Regression model is implement through PySpark's Mllib module. It sets a workflow possible within our own cluster that can be repeated into other parts. It is based off of a sample solution with alterations in style and scale. 

**Part 2**

Part 2 showcases how a Logistic Regression model could be used to classify symptoms of future heart failure among citizens from Framingham, Massachusettes; it isolate on which features stand the most influential for the prediction. Use of common evaluators from a confusion matrix measures the model's performance and a ranking of column importance is outputted. 

A sample [solution housed here](https://www.kaggle.com/code/neisha/heart-disease-prediction-using-logistic-regression) was leveraged to kick the project off and build off of the modularized process from Part 1. Native sample code followed a pandas process and was converted to PySpark to a necessary basic degree. More information resides in the sub-directory `src/part_2`. 


**Part 3**



**Part 4**




# Cloud Git Repo Location and Setup

Git clone this project within your Fordham Big Data Computing Spark setup cluster into a root-user file path of `root@manager:/spark-examples/spark-project2/big_data_project_2/` where `big_data_project_2` in this path is the cloned repo. 

During this process, you should also run the `start.sh` script located at the top directory of `/spark-examples`. This must be done first before executing any of our code. 

For every shell script, allow elevated access of `chmod +x <FILE NAME>`a

