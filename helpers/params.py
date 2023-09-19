"""
docstring
"""
import pandas as pd

# Files parameters
LOG_FILE_PATH = "./logs/churn_library.log"
BANK_DATA_PTH = "./data/bank_data.csv"

# Folders parameters
EDA_OUTPUT_PTH = 'images/eda/'
MODELS_OUTPUT_PTH = 'models/'
RESULTS_OUTPUT_PTH = 'images/results/'

# Figures parameters
DF_HEAD_NAME = 'dataframe_head'
DF_HEAD_TITLE = 'Dataframe head'
DF_SHAPE_NAME = 'dataframe_shape'
DF_SHAPE_TITLE = 'Dataframe shape'
DF_NULL_COUNT_NAME = 'dataframe_head'
DF_NULL_COUNT_TITLE = 'Dataframe head'
DF_DESCRIBE_NAME = 'dataframe_shape'
DF_DESCRIBE_TITLE = 'Dataframe shape'
FIG_CHURN_DIST_NAME = 'churn_distribution'
FIG_CHURN_DIST_TITLE = 'Churn distribution'
FIG_CUST_AGE_DIST_NAME = 'customer_age_distribution'
FIG_CUST_AGE_DIST_TITLE = 'Customer age distribution'
FIG_TOT_TRANS_DIST_NAME = 'total_transaction_distribution'
FIG_TOT_TRANS_DIST_TITLE = 'Total transaction distribution'
FIG_MARIT_STAT_DIST_NAME = 'maritial_status_distribution'
FIG_MARIT_STAT_DIST_TITLE = 'Maritial status distribution'
FIG_HEATMAP_NAME = 'heatmap'
FIG_HEATMAP_TITLE = 'Heatmap'
FIG_ROC_CURVE_NAME = 'roc_curve_result'
FIG_ROC_CURVE_TITLE = 'ROC curve comparison'
FIG_LOGISTIC_REPORT_NAME = 'logistic_results'
FIG_LOGISTIC_REPORT_TITLE = 'Logistic Regression'
FIG_RANDOM_FOREST_REPORT_NAME = 'rf_results'
FIG_RANDOM_FOREST_REPORT_TITLE = 'Random Forest'
FIG_FEAT_IMPOR_BAR_NAME = 'feature_importances'
FIG_FEAT_IMPOR_BAR_TITLE = 'Feature importances Random Forest'
FIG_FEAT_IMPOR_EXPL_NAME = 'tree_explainer'
FIG_FEAT_IMPOR_EXPL_TITLE = 'Tree explainer Random Forest'

# Dataframe parameters
REMOVE_COLUMN_1 = 'Unnamed: 0'
REMOVE_COLUMN_2 = 'CLIENTNUM'
DEPENDENT_COLUMN = "Attrition_Flag"
DEPENDENT_CATEGORY = "Attrited Customer"
TEST_SIZE = 0.3
RESPONSE = "churn"

# Models parameters
LOGISTIC_REGRESSION_NAME = 'logistic_model'
RANDOM_FOREST_NAME = 'rfc_model'



# Parameters for tests
IMPORT_DATA_TESTS = ["./data/bank_data.csv", "data.csv", "bank_data.csv"]
DATA_TESTS = [
    pd.read_csv("./data/bank_data.csv"),
    pd.DataFrame(data={}),
    3,
    'test']
EDA_OUTPUT_PTH_TESTS = [
    'images/eda/',
    'eda/',
    1]
CATEGORY_LST_TESTS = [[
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
], [
    'Other',
    'test',
    'any'              
], [
    3,
    'test',
    True
]]
RESPONSE_TESTS = ['Churn', None, 'any', 'test']
SIZE_TESTS = [0.3, 0.5, 0.8, 4, 15, 'F']
DEPENDENT_COLUMN_TESTS = ["Attrition_Flag", "test", None, 4]
DEPENDENT_CATEGORY_TESTS = ["Attrited Customer", "test", None, 4]
