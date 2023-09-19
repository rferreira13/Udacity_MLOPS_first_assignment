"""
Module with class EDA

Class EDA handle exploratory data analyisis process

Author: Rafael Ferreira
Date: September 2023
"""
from typing import List
import pandas as pd

class EDA:
    """
    EDA (Exploratory Data Analysis) Class for performing basic data analysis tasks.

    Parameters:
        data (pd.DataFrame): The DataFrame to perform EDA on.
        Must be a not empty Pandas Dataframe

    Class functions:
        - describe -> Returns pd.DataFrame containing dataframe description
        - shape -> Returns pd.DataFrame containing number of rows and columns on dataframe
        - null_count -> Returns pd.DataFrame containing number of null values per column
        - category_columns -> Returns list of strings containing category columns
        - numerical_columns -> Returns list of strings containing category columns

    Usage example:
    ```
    data = pd.read_csv("data.csv") #Load any pd.DataFrame object
    eda = EDA(data) #Instantiate EDA object with loadad dataframe
    print(eda.data) #Print loaded data
    print(eda.describe()) #Print data description
    print(eda.shape()) #Print data shape
    print(eda.null_count()) #Print number of null values per column
    print(eda.category_columns()) #Print categorical columns
    print(eda.numerical_columns()) #Print numerical columns
    ```
    """
    CATEGORY_DTYPES = ['object', 'category']
    NUMERIC_DTYPES = ['int64', 'float64', 'int32', 'float32']

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the EDA class with a Pandas DataFrame.

        Args:
            data(pd.DataFrame): The DataFrame to perform EDA on.

        Raises:
            TypeError: If the input is not a DataFrame.
            ValueError: If the DataFrame is empty.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a DataFrame.")
        if data.empty:
            raise ValueError("DataFrame must contain data.")

        self.__data = data

    def describe(self) -> pd.DataFrame:
        """
        Get various summary statistics for the DataFrame.

        Returns:
            A DataFrame containing summary statistics.
        """
        return self.__data.describe().reset_index().rename(columns={'index': 'Statistic method'})

    def shape(self) -> pd.DataFrame:
        """Get the shape of the DataFrame.

        Returns:
            A DataFrame containing the number of rows and columns.
        """
        return pd.DataFrame({
            'Number of Rows': [self.__data.shape[0]],
            'Number of Columns': [self.__data.shape[1]]
        })

    def null_count(self) -> pd.DataFrame:
        """Count the number of null values for each column.

        Returns:
            A DataFrame containing the number of null values for each column.
        """
        null_values = self.__data.isnull().sum().reset_index(name='null values')
        return null_values.rename(columns={'index': 'Column name'})

    def category_columns(self) -> List[str]:
        """Get the names of the categorical columns.

        Returns:
            A list containing the names of the categorical columns.
        """
        return self.__data.select_dtypes(include=self.CATEGORY_DTYPES).columns.tolist()

    def numerical_columns(self) -> List[str]:
        """Get the names of the numerical columns.

        Returns:
            A list containing the names of the numerical columns.
        """
        return self.__data.select_dtypes(include=self.NUMERIC_DTYPES).columns.tolist()

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the DataFrame being analyzed.

        Returns:
            The DataFrame being analyzed.
        """
        return self.__data

if __name__ == '__main__':
    # Load dataframe
    df = pd.read_csv(r"./data/bank_data.csv")
    # Dataframe holding statistics
    print(EDA(data=df).describe())
    # List containing numerical columns
    print(EDA(data=df).numerical_columns())
    # List containing categorical columns
    print(EDA(data=df).category_columns())
    # Dataframe counting null values on dataframe
    print(EDA(data=df).null_count())
