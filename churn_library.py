"""
The churn_library.py is a library of functions
to find customers who are likely to churn.

Author: Rafael Ferreira
Date: September 2023
"""
import os
from typing import Optional
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from helpers.figure import Figures
from helpers.eda import EDA
from helpers.feature_engineering import FeatureEngineering
import helpers.params as p

os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(path: str) -> pd.DataFrame:
    """
    Load data from the CSV file and return as a DataFrame.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the specified file is not found.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If an error occurred while parsing the CSV file.
        ValueError: If the input path is not valid.

    Usage:
    ```
    df = import_data("data.csv")
    ```
    """
    try:
        return pd.read_csv(path)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File {path} not found.") from exc

    except pd.errors.EmptyDataError as exc:
        logging.error("ValueError")
        raise ValueError("The CSV file is empty.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError("Error occurred while parsing the CSV file.") from exc

    except ValueError as exc:
        raise ValueError("You must pass a valid entry.") from exc

def perform_eda(eda_data:pd.DataFrame, eda_output_pth:str) -> None:
    """
    Pipeline to save EDA process

    Args:
        eda_data (pd.DataFrame): Instantiated EDA object with dataframe
        eda_output_pth (str): Instantiated Figures object with setted eda folder

    Returns:
        None
        Will save figures on setted eda folder

    Raises:
        TypeError: If eda_object is not EDA type.
        TypeErrorr: If figure_object is not Figures type.

    Usage:
    ```
    df = import_data("data.csv")
    eda_output_pth = 'images'
    perform_eda(eda_data = df, eda_output_pth = eda_output_pth)
    ```
    """

    # Save EDA Figures
    churn_eda = EDA(eda_data) # Instantiate object to handle EDA process

    #Select name of the folder to save EDA figures
    Figures.set_eda_folder(folder=eda_output_pth)

    # Save dataframe head
    Figures.table_fig(
        dataframe = churn_eda.data.head(),
        fig_name = p.DF_HEAD_NAME,
        title = p.DF_HEAD_TITLE
    )

    # Save dataframe shape
    Figures.table_fig(
        dataframe = churn_eda.shape(),
        fig_name = p.DF_SHAPE_NAME,
        title = p.DF_SHAPE_TITLE
    )

    # Save dataframe null values count
    Figures.table_fig(
        dataframe = churn_eda.null_count(),
        fig_name = p.DF_NULL_COUNT_NAME,
        title = p.DF_NULL_COUNT_TITLE
    )

    # Save dataframe statistics summary
    Figures.table_fig(
        dataframe = churn_eda.describe(),
        fig_name = p.DF_DESCRIBE_NAME,
        title = p.DF_DESCRIBE_TITLE
    )

    # Save histogram from Attrition_Flag
    Figures.histogram_fig(
        series = churn_eda.data.Attrition_Flag,
        fig_name = p.FIG_CHURN_DIST_NAME,
        title = p.FIG_CHURN_DIST_TITLE
    )

    # Save histogram from Customer_Age
    Figures.histogram_fig(
        series = churn_eda.data.Customer_Age,
        fig_name = p.FIG_CUST_AGE_DIST_NAME,
        title = p.FIG_CUST_AGE_DIST_TITLE
    )

    # Save histogram from Total_Trans_Ct - Adding kernel density estimate
    Figures.histogram_fig(
        series = churn_eda.data.Total_Trans_Ct,
        fig_name = p.FIG_TOT_TRANS_DIST_NAME,
        title = p.FIG_TOT_TRANS_DIST_TITLE,
        stat='density',
        kde=True
    )

    # Save histogram from Marital_Status - Adding data normalization (density flag)
    Figures.histogram_fig(
        series = churn_eda.data.Marital_Status,
        fig_name = p.FIG_MARIT_STAT_DIST_NAME,
        title = p.FIG_MARIT_STAT_DIST_TITLE,
        stat='density'
    )

    # Save dataframe heatmap
    Figures.heatmap_fig(
        dataframe = churn_eda.data,
        fig_name = p.FIG_HEATMAP_NAME,
        title = p.FIG_HEATMAP_TITLE
    )

def perform_feature_engineering(
        feat_data: pd.DataFrame,
        size: float,
        dependent_column: str,
        dependent_category: Optional[str],
        response: Optional[str]
    ):
    """
    Pipeline to perform feature engineering

    Args:
        feat_data (pd.DataFrame): Data to perform feature engineering.
        size (float): Size of train test split.
        dependent_column (str): Selected column as targert.
        dependent_category (str, optional): If selected column as targert
            is categorical, selects the target category.
        response (str, optional): Optional name for response data.

    Returns:
        X_train, X_test, y_train, y_test
        Where:
            X_train (pd.DataFrame)
            X_test (pd.DataFrame)
            y_train (pd.Series)
            y_test (pd.Series)

    Raises:
        TypeError: If the input feat_data is not a DataFrame.
        ValueError: If the DataFrame is empty.
        KeyError: If argument column not in data columns.
        TypeError: If argument column is not numerical and argument category is not declared.
        KeyError: If argument category not in selected response column.

    Usage:
    ```
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        feat_data = data,
        size: 0.3,
        dependent_column: "Attrition_Flag",
        dependent_category: "Attrited Customer",
        response: "churn"
    )
    ```
    """
    feat_obj = FeatureEngineering(feat_data)
    eda_obj = EDA(feat_data)
    # Selecting column to extract response
    dependent_data = feat_obj.select_response(
        column = dependent_column,
        category = dependent_category
        )

    feat_obj.remove_columns(columns = [dependent_column])

    # Add response data to dataframe with selected response name
    feat_obj.add_column(
        column_data = dependent_data,
        column_name = response
    )

    # Encode categorical data
    feat_obj.encode(
        category_lst = eda_obj.category_columns(),
        response = response
    )

    feat_obj.remove_columns(columns = eda_obj.category_columns())

    target = feat_obj.data[response]
    feat_obj.remove_columns(columns = [response])
    predictors = feat_obj.data

    return train_test_split(predictors, target, test_size = size, random_state = 42)

def classification_report_image(
        output_pth:str,
        test_data:tuple,
        train_data:tuple,
        models:list,
        **kwargs
    ) -> None:
    """
    Save models classification report

    produces classification report for training and testing results 
    and stores report as image in output path

    Args:
        output_pth (str): Path to save images
        test_data (tuple(pd.DataFrame, pd.Series)):
            Tuple containing (features_test data, y_test_data) in that order
        train_data (tuple(pd.DataFrame, pd.Series)):
            Tuple containing (features_train data, y_train_data) in that order
        models (list): List containing models to build classification report

    Returns:
        None
        Will save figures on output path

    Raises:
        TypeError: If eda_object is not EDA type.
        TypeErrorr: If figure_object is not Figures type.

    Usage:
    ```
    df = import_data("data.csv")
    ChurnEDA = EDA(df)
    FigureManager = Figures
    FigureManager.set_eda_folder(folder='images')
    perform_eda(ChurnEDA, FigureManager)
    ```
    """
    Figures.set_eda_folder(folder=output_pth)

    features_test, y_test_data = test_data
    features_train, y_train_data = train_data

    for i, model in enumerate(models):
        Figures.classification_report_fig(
            y_test_data = y_test_data,
            y_test_pred = model.predict(features_test),
            y_train_data = y_train_data,
            y_train_pred = model.predict(features_train),
            fig_name = kwargs.get('names')[i] if kwargs.get('names') else None,
            title = kwargs.get('titles')[i] if kwargs.get('titles') else None
        )

def feature_importance_plot(
        model: RandomForestClassifier,
        features_test: pd.DataFrame,
        output_pth: str,
        **kwargs
    ):
    """
    docstring
    creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                 None
    """
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("Parameter model must be RandomForestClassifier")
    # Set folder to save figures
    Figures.set_eda_folder(folder=output_pth)
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [features_test.columns[i] for i in indices]

    Figures.bar_fig(
        values = importances[indices],
        names = names,
        fig_name = kwargs.get('bar_fig_name'),
        title = kwargs.get('bar_fig_title')
    )

    Figures.tree_explainer_fig(
        model = model,
        x_test = features_test,
        fig_name = kwargs.get('explainer_fig_name'),
        title = kwargs.get('explainer_fig_title')
    )

def train_models(
        features_train: pd.DataFrame,
        features_test: pd.DataFrame,
        target_train: pd.Series,
        target_test: pd.Series,
        train_the_model: bool
    ):
    """
    train, store model results: images + scores, and store models

    Args:
        features_train (pd.DataFrame): The data features to train models.
        features_test (pd.DataFrame): The data features to test models.
        target_train (pd.Series): The target data to train models.
        target_test (pd.Series): The target data to test models.
        train_the_model (bool): If True will train models,
            else will load from pre trained models

    Returns:
        None
        Will train models and store results

    Raises:
        TypeError: If output path is not string type.
        ValueError: If output path is not a valid path.
        TypeErrorr: If features_train is not pandas DataFrame.
        TypeErrorr: If target_train is not pandas Series.
        TypeErrorr: If lr_name is not string type.
        TypeErrorr: If rf_name is not string type.
        FileNotFoundError: If lr_name file at folder doesn't exist.
        FileNotFoundError: If rf_name file at folder doesn't exist.
        TypeErrorr: If lr_name file exists but is not LogisticRegression.
        TypeErrorr: If rf_name file exists but is not RandomForestClassifier.
        TypeError: If argument folder is not type string.
        ValueError: If argument folder is not a valid path.

    Usage:
    ```
    train_models(
        features_train: X_data_train,
        features_test: X_data_test,
        target_train: y_data_train,
        target_test: y_data_test,
        train_the_model: False
    )
    ```
    """

    if train_the_model:

        train_store_models(
            features_train = features_train,
            target_train = target_train,
            output_pth = p.MODELS_OUTPUT_PTH,
            lr_name = p.LOGISTIC_REGRESSION_NAME,
            rf_name = p.RANDOM_FOREST_NAME
        )

    rfc_model, lr_model = load_models(
        folder = p.MODELS_OUTPUT_PTH,
        lr_name = p.LOGISTIC_REGRESSION_NAME,
        rf_name = p.RANDOM_FOREST_NAME
    )

    store_roc_curves(
        models = [rfc_model, lr_model],
        features_test = features_test,
        target_test = target_test,
        output_pth = p.RESULTS_OUTPUT_PTH,
        fig_name = p.FIG_ROC_CURVE_NAME,
        title = p.FIG_ROC_CURVE_TITLE)

    classification_report_image(
        output_pth = p.RESULTS_OUTPUT_PTH,
        test_data = (features_test, target_test),
        train_data = (features_train, target_train),
        models = [lr_model, rfc_model],
        names = [p.FIG_LOGISTIC_REPORT_NAME, p.FIG_RANDOM_FOREST_REPORT_NAME],
        titles = [p.FIG_LOGISTIC_REPORT_TITLE, p.FIG_RANDOM_FOREST_REPORT_TITLE])

    feature_importance_plot(
        model = rfc_model,
        features_test = features_test,
        output_pth = p.RESULTS_OUTPUT_PTH,
        bar_fig_name = p.FIG_FEAT_IMPOR_BAR_NAME,
        bar_fig_title = p.FIG_FEAT_IMPOR_BAR_TITLE,
        explainer_fig_name = p.FIG_FEAT_IMPOR_EXPL_NAME,
        explainer_fig_title = p.FIG_FEAT_IMPOR_EXPL_TITLE
    )

def store_roc_curves(
        models: list,
        features_test: pd.DataFrame,
        target_test: pd.Series,
        output_pth: str = '',
        **kwargs
    ) -> None:
    """
    Save ROC curves

    If fig_name is not declared will only display figure

    Args:
        models (list): list with models.
        features_test (pd.DataFrame): features to test models, must have
            same columns size as data used to train models.
        target_test (pd.Series): targets to test models, must have
            same rows size as data used to train models.
        output_pth (str): Path to ROC curves figure. Defaults to current folder.
        fig_name (str, optional): Name of the figure for saving. Defaults to None.
        title (str, optional): Title of the figure. Defaults to None.

    Returns:
        None
        If fig_name is declared will save figure.
        If fig_name is not declared will only display figure.


    Raises:
        TypeError: If argument folder is not type string.
        ValueError: If argument folder is not a valid path.

    Usage:
    ```
    store_roc_curves(
        models: [rfc_model, lr_model],
        features_test: X_test,
        target_test: y_test,
        output_pth: str = 'images/results/',
        fig_name = 'roc_curve_result',
        title = 'ROC curves'
    )
    ```
    """

    Figures.set_eda_folder(output_pth)

    Figures.roc_curve_fig(
        models = models,
        x_test = features_test,
        y_test = target_test,
        **kwargs
    )

def load_models(folder: str, lr_name: str, rf_name: str) -> tuple:
    """
    Load trained logistic regression and random forest models

    Args:
        folder (str): Folder where the models are stored.
        lr_name (str): Name to load logistic regression model
        rf_name (str): Name to load random forest model

    Returns:
        Tuple: (RandomForestClassifier, LogisticRegression)
        Will return a tuple with RandomForestClassifier
        and LogisticRegression in that order

    Raises:
        FileNotFoundError: If lr_name file at folder doesn't exist.
        FileNotFoundError: If rf_name file at folder doesn't exist.
        TypeErrorr: If lr_name file exists but is not LogisticRegression.
        TypeErrorr: If rf_name file exists but is not RandomForestClassifier.

    Usage:
    ```
    rfc_model, lr_model = load_models(
        folder = 'models/',
        lr_name = 'logistic_model',
        rf_name = 'rfc_model')
    print(rfc_model)

    ```
    """
    if not os.path.exists(f"{folder}{rf_name}.pkl"):
        raise FileNotFoundError(f"File {folder}{rf_name}.pkl does not exist")

    if not os.path.exists(f"{folder}{lr_name}.pkl"):
        raise FileNotFoundError(f"File {folder}{lr_name}.pkl does not exist")

    random_forest = joblib.load(f"{folder}{rf_name}.pkl")

    if not isinstance(random_forest, RandomForestClassifier):
        raise TypeError(f"File {folder}{rf_name}.pkl exists, but is not RandomForestClassifier")

    logistic_regression = joblib.load(f"{folder}{lr_name}.pkl")

    if not isinstance(logistic_regression, LogisticRegression):
        raise TypeError(f"File {folder}{lr_name}.pkl exists, but is not LogisticRegression")

    return random_forest, logistic_regression

def train_store_models(
        features_train: pd.DataFrame,
        target_train: pd.Series,
        output_pth: str,
        lr_name: str,
        rf_name: str
    ):
    """
    Train and store logistic regression and random forest models

    Args:
        features_train (pd.DataFrame): The DataFrame to perform EDA on.
        target_train (pd.Series): The DataFrame to perform EDA on.
        output_pth (str): Path to store models
        lr_name (str): Name to save logistic regression model
        rf_name (str): Name to save random forest model

    Returns:
        None
        Will train and store, at given output path logistic regression and random forest models

    Raises:
        TypeError: If output path is not string type.
        ValueError: If output path is not a valid path.
        TypeErrorr: If features_train is not pandas DataFrame.
        TypeErrorr: If target_train is not pandas Series.
        TypeErrorr: If lr_name is not string type.
        TypeErrorr: If rf_name is not string type.

    Usage:
    ```
    features_train = pd.DataFrame(data={
        'a':[1,5,2,5,9,0,3,2,1,7,9],
        'b':[6,8,2,7,1,0,2,4,9,6,1],
        'c':[4,1,3,9,0,0,3,2,6,8,5]})
    target_train = pd.Series([3,2,1,6,8,9,2,1,4,7,9])
    output_pth = ''
    lr_name = 'logistic'
    rf_name = 'random'
    train_store_models(features_train,target_train,output_pth,lr_name,rf_name)
    ```
    """
    if not isinstance(output_pth, str):
        raise TypeError("Parameter output_pth must be string type.")
    if not os.path.isdir(output_pth) and output_pth!='':
        raise ValueError("Parameter output_pth is not a valid path.")
    if not isinstance(features_train, pd.DataFrame):
        raise TypeError("Parameter features_train must be pandas dataframe.")
    if not isinstance(target_train, pd.Series):
        raise TypeError("Parameter target_train must be pandas series.")
    if not isinstance(lr_name, str):
        raise TypeError("Parameter lr_name must be string type.")
    if not isinstance(rf_name, str):
        raise TypeError("Parameter rf_name must be string type.")

    if os.path.isdir(output_pth) and not output_pth.endswith('/'):
        folder = folder + '/'

    #Prepare random forest model
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    #Prepare Logistic Regression model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Train models
    cv_rfc.fit(features_train, target_train)
    lrc.fit(features_train, target_train)

    # Store models
    joblib.dump(cv_rfc.best_estimator_, f"{output_pth}{rf_name}.pkl")
    joblib.dump(lrc, f"{output_pth}{lr_name}.pkl")


if __name__ == '__main__':

    data = import_data(p.BANK_DATA_PTH)

    #Perform EDA
    perform_eda(eda_data = data, eda_output_pth = p.EDA_OUTPUT_PTH)

    # Create Feature Engineering object and remove inappropriate columns
    clean_data = FeatureEngineering(data)
    clean_data.remove_columns([p.REMOVE_COLUMN_1, p.REMOVE_COLUMN_2])

    #Perform Feature Engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        feat_data = clean_data.data,
        size = p.TEST_SIZE,
        dependent_column = p.DEPENDENT_COLUMN,
        dependent_category = p.DEPENDENT_CATEGORY,
        response = p.RESPONSE)

    train_models(
        features_train = X_train,
        features_test = X_test,
        target_train = y_train,
        target_test = y_test,
        train_the_model = False
    )
