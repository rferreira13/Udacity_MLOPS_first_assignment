"""
Module with class FeatureEngineering

Class FeatureEngineering handle feature engineering process

Author: Rafael Ferreira
Date: September 2023
"""
from typing import Optional, List
import pandas as pd

class FeatureEngineering:
    """
    FeatureEngineering Class for preparing training data.

    Attributes:
        data (pd.DataFrame): The DataFrame with the features.

    Class Methods:
    - encode(category_lst: List[str], response: Optional[str] = None) -> None
    - remove_columns(columns: List[str]) -> None
    - add_column(column_data: pd.Series, column_name: str) -> None
    - select_response(column: str, category: Optional[str] = None) -> pd.Series

    Property:
    - data -> pd.DataFrame

    Usage example:
    ```
    feature_engineering = FeatureEngineering(data)
    feature_engineering.encode(['column_1', 'column_2'], 'response_column')
    print(feature_engineering.data)
    ```
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the FeatureEngineering class with a DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to perform feature engineering on.

        Raises:
            TypeError: If the input is not a DataFrame.
            ValueError: If the DataFrame is empty.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a DataFrame.")
        if data.empty:
            raise ValueError("DataFrame must contain data.")

        self.__data = data

    def encode(self, category_lst: List[str], response: Optional[str] = None) -> None:
        """
        Encode categorical columns with the proportion of a response variable.

        Args:
            category_lst (List[str]): List of categorical columns to encode.
            response (Optional[str]): string of response name [optional argument that could
            be used for naming variables or index y column]

        Raises:
            KeyError: If any column in category_lst is not in the DataFrame.
            KeyError: If 'Churn' or response is not in the DataFrame.
        """
        response_prefix = '' if not response else f'_{response}'

        if not set(category_lst).issubset(set(self.__data.columns)):
            raise KeyError("Parameter category_lst must be a subset from data columns.")

        if response:
            if response not in self.__data.columns:
                raise KeyError(f"Column {response} not found in DataFrame.")

            for categorical_col in category_lst:

                categorical_groups = self.__data.groupby(categorical_col).mean()[response]
                new_col_name = f"{categorical_col}{response_prefix}"
                self.__data[new_col_name] = self.__data[categorical_col].map(categorical_groups)

        else:

            hot_encode = pd.get_dummies(self.__data[category_lst]).reset_index()
            self.__data = self.__data.reset_index().merge(hot_encode).drop(columns=['index'])

    def remove_columns(self, columns: List[str]) -> None:
        """
        Remove selected columns from dataframe

        Args:
            columns (List[str]): List containing columns to remove from dataframe

        Raises:
            TypeError: If columns to remove is not from type list
            KeyError: If columns list is not a subset from datacolumns
        """
        if not isinstance(columns, list):
            raise TypeError("Columns must be type list")
        if not set(columns).issubset(set(self.__data.columns)):
            raise KeyError("All columns must be in dataframe columns")
        self.__data.drop(columns=columns, inplace=True)

    def add_column(self, column_data: pd.Series, column_name: str) -> None:
        """
        Add column to dataframe

        Args:
            column_data (pd.Series): Data to insert
            column_name (str): Name to be given to new column

        Raises:
            TypeError: If columns to remove is not from type list
            KeyError: If columns list is not a subset from datacolumns
        """
        if not isinstance(column_data, pd.Series):
            raise TypeError("Argument column_data must be Pandas Series type")
        if self.__data.shape[0] != len(column_data):
            raise ValueError("Length of column_data must match dataframe")
        self.__data[column_name] = column_data

    def select_response(self, column: str, category: Optional[str] = None) -> pd.Series:
        """
        Extract response data from dataframe

        Args:
            column (str): Name of the column selected as response
            category (Optional[str]): If response is not numerical, category is required,
            when declared, category must be present in response column

        Raises:
            KeyError: If argument column not in data columns.
            TypeError: If argument column is not numerical and argument category is not declared.
            KeyError: If argument category not in selected response column.
        """

        if column not in self.__data.columns:
            raise KeyError(f"Column {column} not found in DataFrame.")

        if self.__data[column].dtype not in ['int64', 'float64', 'int32', 'float32']:
            if not category:
                raise TypeError(f"Categorical response {column} require argument 'category'")

            if category not in self.__data[column].unique():
                raise KeyError(f"Category {category} not in response column {column}")

            return pd.get_dummies(self.__data[column])[category]

        return self.__data[column]

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the dataframe.

        Returns:
            pd.DataFrame: The dataframe.
        """
        return self.__data

if __name__ == '__main__':
    # Load dataframe
    # df = pd.read_csv(r"./data/bank_data.csv")
    df = pd.DataFrame(data = {
        'column 1':['a','b','a','a','b','a'],
        'column 2':[1,2,3,1,2,3],
        'column 3':['c','c','c','d','d','d']
    })
    # Instantiate object
    feature_engineering = FeatureEngineering(df)
    # Save response column selecting desired category
    response_data = feature_engineering.select_response(
        column = 'column 1',
        category = 'a'
        )
    # Remove response column from dataframe
    feature_engineering.remove_columns(columns = ['column 1'])
    # Add engineered response data to dataframe
    feature_engineering.add_column(
        column_data = response_data,
        column_name = 'response_name'
    )
    # Select categorical columns to encode in function of engineered column
    feature_engineering.encode(
        category_lst = ['column 3'],
        response = 'response_name'
    )
    # Remove selected categorical data
    feature_engineering.remove_columns(columns = ['column 3'])
    # Print dataframe
    print(feature_engineering.data)
