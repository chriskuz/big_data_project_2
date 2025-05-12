# Random Forest and Decision Trees to Predict Income

## General 
This part of the project was the same as part 3, as in we were using the same data to predict the same information, whether an individual's income is above or below $50K per year based on demographic and employment attributes. The only thing that was different for this part was that instead of using a Logistic Regression model, we used a Random Forest Classifier and a Decision Tree's Classifier. The dataset contains both numerical and categorical features. Categorical features were transformed using StringIndexer and OneHotEncoder, and numerical features were combined with the encoded columns via a VectorAssembler into a final features vector used for model training. The target column (income) was cleaned and indexed into a binary label.

The code splits the dataset into training and testing sets, fits a logistic regression model, and evaluates it using standard classification metrics such as accuracy, F1 score, precision, and recall. 

No true nulls existed in the dataset, but the code includes logic to check for placeholder values (?) often used in this dataset to indicate missing data. Because none were found in this specific instance, no rows were dropped.

## Data

[The data can be found by clicking here](https://archive.ics.uci.edu/dataset/20/census+income). A `get_data.sh` script exists for this part of the project which will download the data directly from the website and place it into the correct data directory.

## Execution

Provide elevated scurity acces within the repository for the shell scripts and run the `get_data.sh` to download the data. Then run the `census_models.sh` script which will execute the `census_models.py` script to run the model and generate command line results.