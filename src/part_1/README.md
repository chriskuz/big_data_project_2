Goal: the goal of this is to use a Spark ML flow to determine the probability of each type of toxicity for each comments

# Logistic Regression to Extract Probabilies

This part showcases how a Logistic regression Model could be used in PySpark to extract probabilities of measuring toxicness of human written comments. The code leverages the MLlib package in PySpark to prepare the data with TFIDF workflow and feed it into a prepped `LogiSticRegression()` model class. 

[The data can be found by clicking here].()