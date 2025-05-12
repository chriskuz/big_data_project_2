# Logistic Regression to Predict Income

## General 

This part of the project demonstrates how a logistic regression model can be applied using PySpark MLlib to predict whether an individual's income is above or below $50K per year based on demographic and employment attributes from the UCI Census dataset. The code leverages the MLlib package in PySpark to leverage a similar `LogiSticRegression()` flow as Part 1. The implementation mirrors the general structure of Part 1, but includes additional steps for categorical variable handling, which were not necessary in Part 2. The dataset contains both numerical and categorical features. Categorical features were transformed using StringIndexer and OneHotEncoder, and numerical features were combined with the encoded columns via a VectorAssembler into a final features vector used for model training. The target column (income) was cleaned and indexed into a binary label.

The code splits the dataset into training and testing sets, fits a logistic regression model, and evaluates it using standard classification metrics such as accuracy, F1 score, precision, and recall. 

No true nulls existed in the dataset, but the code includes logic to check for placeholder values (?) often used in this dataset to indicate missing data. Because none were found in this specific instance, no rows were dropped.

## Data

[The data can be found by clicking here](https://archive.ics.uci.edu/dataset/20/census+income). A `get_data.sh` script exists for this part of the project which will download the data directly from the website and place it into the correct data directory.

## Execution

Provide elevated scurity acces within the repository for the shell scripts and run the `get_data.sh` to download the data. Then run the `census_income.sh` script which will execute the `census_income.py` script to run the model and generate command line results.