# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Install requirements

	pip install pandas==1.1.5
	pip install scikit-learn==0.24.2
	pip install shap==0.41.0
	pip install matplotlib==3.3.4
	pip install seaborn==0.11.2
	pip install pylint==2.13.9
	pip install autopep8==2.0.4
	pip install pytest==7.0.1
	pip install ipython==7.16.3

You may also run

    pip install -r requirements.txt

## Project Description
This is a project to identify credit card customers that are most likely to churn, implementing best coding practices.

## Files and data description
- churn_library.py - The churn_library.py is a library of functions to find customers who are likely to churn. 

- churn_script_logging_and_tests.py - The churn_script_logging_and_tests.py contain unit tests for the churn_library.py functions

- conftest.py - Sets pytest logging file.

- pytest.ini - Determines pytest to run churn_script_logging_and_tests.py.

- data/bank_data.csv - Contains data from credit card customers.

- helpers/eda.py - Contains class EDA, for performing basic data analysis tasks. It contains methods:
    - describe - Returns pd.DataFrame containing dataframe description
    - shape - Returns pd.DataFrame containing number of rows and columns on dataframe
    - null_count - Returns pd.DataFrame containing number of null values per column
    - category_columns - Returns list of strings containing category columns
    - numerical_columns - Returns list of strings containing category columns

- helpers/feature_engineering.py - Contains class FeatureEngineering, for performing feature engineering tasks. It contains methods:
    - encode - Encode categorical columns with the proportion of a response variable.
    - remove_columns - Remove selected columns from dataframe
    - add_column - Add column to dataframe
    - select_response - Extract response data from dataframe

- helpers/figure.py - Contains class Figures, for assisting in the EDA and model performance process. It contains methods:
    - set_eda_folder - Set the folder path for saving figures.
    - table_fig - Create and display/save a table figure from a DataFrame.
    - histogram_fig - Create and display/save a histogram figure from a Series.
    - heatmap_fig - Create and display/save a heatmap figure from a DataFrame.
    - roc_curve_fig - Create and display/save a ROC figure from a list of models, based on test data.
    - classification_report_fig - Create and display/save a classification report figure based on train and test data.
    - bar_fig - Create and display/save a bar figure based on alist of names and a array of values.
    - tree_explainer_fig - Create and display/save a explainer for a random forest model.

- helpers/params.py - Contains constants for churn_library.py and churn_script_logging_and_tests.py.


## Running Files
Run churn_library.py with the command below. It will train save EDA figures, perform feature engineering, train models and save performance metrics.

    ipython churn_library.py

Run churn_script_logging_and_tests.py with the command below. It will test functions and save logs in log file.

    ipython churn_script_logging_and_tests.py

You may as well run the command:

    pytest



