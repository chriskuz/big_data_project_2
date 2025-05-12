https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset?select=framingham.csv

Data


https://www.kaggle.com/code/neisha/heart-disease-prediction-using-logistic-regression




# Logistic Regression to Determine Heart Disease

## General

This part showcases how a Logistic regression Model could be used in PySpark to classify late-stage heart disease/failure. The code leverages the MLlib package in PySpark to leverage a similar `LogiSticRegression()` flow as Part 1. However, the values within the dataset were all numeric based natively and required little transformation apart from vectorization with a `VectorAssembler()`. Some nulls were present and blanketly dropped due to lack of ethics in making up data for predicting a health concern. The program was based off a sample solution ([found here](https://www.kaggle.com/code/neisha/heart-disease-prediction-using-logistic-regression)) built through pandas and providing an in depth study with visuals. For this report, the basics of the project to produce a model, discuss the null composition, and provide a KPI-based ranking of feature importance via coefficient weighting provided within the Logistic Regression package. 

Common confusion matrix based scoring metrics were leveraged including accuracy, f1 score, precision, and recall. The training data was split to identify the success of a model. 

Overall, a basic concept of how to implement a similar model workflow built from Part 1 is introduced and provided as an example for how one can progress with their modeling within a cluster network. 

## Data

[The data can be found by clicking here](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset?select=framingham.csv). A `get_data.sh` script exists for this part of the project as the Kaggle host-domain for this dataset provides a requestable link to directly download. This is where it is important to be sure your Kaggle API is set properly into the home directory. Please refer to the documentation at the root README of this repositoryfor an explanation of how to set it up.

## Execution

Provide elevated scurity acces within the repository for the shell scripts and run the `get_data.sh` to download the data pending Kaggle API setup. Then run the `diagnose_heart.sh` script which will execute the `diagnose_heart.py` script to run the model and generate command line results. 