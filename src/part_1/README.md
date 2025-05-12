# Logistic Regression to Extract Probabilies

## General

This part showcases how a Logistic regression Model could be used in PySpark to extract probabilities of measuring toxicness of human written comments. The code leverages the MLlib package in PySpark to prepare the data with TFIDF workflow and feed it into a prepped `LogiSticRegression()` model class. The program was based off a sample solution ([found here](https://www.kaggle.com/code/bkarabay/simple-text-classification-with-apache-spark)) identifying probabilities of toxicity surrounding a label class kknown as `toxic`. This prooject compartmentalizes a similar workflow and iterates through all labels of toxic-like descriptions to showcase scalability. 

The testing set is brought in as representation of how additional data can be fed into the workflow. Though, scoring is commonly done with a submission to Kaggle on a `sample` set which is not shown. **Vageuness and ambguity in the project's solution lead to just a showcasing of expansion with no considartions on the labels themselves.** Though a late discovery did note the existence of a `test_labels.csv` which can be used by the user with their own modifications to expand workflow.

## Data

[The data can be found by clicking here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). When downloading the data, there will be multiple files embedded compressed within a zipped folder. You can choose to unpackage the data here or unpackage it upstream. The only datasets we are concerned with are the `train.csv` and the `test.csv` files. These files will be called upon by our `toxic_comment_classifier.py` program. 

The workflow used to push the data to the cloud involved a local copy-paste up into the root directory of the project within a git ignored `data` folder. There is no shell script here that successfully leverages Kaggle to pull this data directly as the dataset is hidden behind a Kaggle authentication protocol. 

## Execution

Provide elevated scurity acces within the repository for the shell scripts and run the `toxic_comment_classifier.sh` to begin the program. The `toxic_comment_classifier.py` file will be executed in a Spark environment after this. 