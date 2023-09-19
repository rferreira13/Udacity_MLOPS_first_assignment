"""
Module with class Figures

Class Figures handle saving figures

Author: Rafael Ferreira
Date: September 2023
"""
import os
import re
import warnings
from typing import List, Optional
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
sns.set()

INVALID_CHARS = r'[<>:"/\\|?*]'

class Figures:
    """
    Figures class for assisting in the EDA and model performance process

    Class Methods:
    - set_eda_folder(folder: str) -> None
    - table_fig(dataframe: pd.DataFrame, fig_name: str, title: str) -> None
    - histogram_fig(series: pd.Series, fig_name: str, title: str, stat: str, kde: bool) -> None
    - heatmap_fig(dataframe: pd.DataFrame, figure_name: str, title: str) -> None
    - roc_curve_fig(
        models:list,
        x_test:pd.DataFrame,
        y_test:pd.Series,
        fig_name: str, optional,
        title: str, optional
    )
    - classification_report_fig(
        y_test_data:pd.Series,
        y_test_pred:np.ndarray,
        y_train_data:pd.Series,
        y_train_pred:np.ndarray,
        fig_name: str, optional,
        title: str, optional
    )
    - bar_fig(
        values:np.ndarray,
        names:List[str],
        fig_name: str, optional,
        title: str, optional
    )
    - tree_explainer_fig(
        model:RandomForestClassifier,
        x_test:pd.DataFrame,
        fig_name: str, optional,
        title: str, optional
    )
    
    Usage:
    ```
    df = pd.read_csv('data.csv')
    Figures.set_eda_folder("figures")
    Figures.table_figure(df.head(), figure_name="table", title="Table Title")
    Figures.histogram_figure(df['column'], figure_name="hist", title="Histogram")
    Figures.heatmap_figure(df, figure_name="heatmap", title="Heatmap")
    ```
    """

    fig_folder = ''

    @classmethod
    def __finalize_fig(cls, fig_name=None, title=None) -> None:
        """
        Add title and either save or show the figure based on the figure_name.

        Args:
            fig_name (str, optional): Name of the figure for saving. Defaults to None.
            title (str, optional): Title of the figure. Defaults to None.
        """
        if title:
            cls.__add_title(title)
        if fig_name:
            cls.__save_fig(fig_name)
        else:
            cls.__show_fig()

    @classmethod
    def __save_fig(cls, fig_name:str) -> None:
        """
        Save the figure as a PNG file.

        Args:
            fig_name (str): Name of the figure to be saved.

        Raises:
            TypeError: If argument fig_name is not type string.
            ValueError: If argument fig_name is not a valid name pattern.
            ValueError: If argument fig_name has more than 40 or 0 characters.
        """
        if not isinstance(fig_name, str):
            raise TypeError("Parameter fig_name must be string type")
        if re.search(INVALID_CHARS, fig_name):
            raise ValueError("Parameter fig_name contains invalid characters")
        if len(fig_name) > 40 or len(fig_name)==0:
            raise ValueError("Parameter must contain more than 0 and less than 40 characters")
        plt.savefig(f"{cls.fig_folder}{fig_name}.png", format='png')
        plt.close()

    @classmethod
    def __show_fig(cls) -> None:
        """Display the figure."""
        plt.show()

    @classmethod
    def __add_title(cls, title:str) -> None:
        """Add a title to the figure."""
        if not isinstance(title, str):
            raise TypeError("Parameter title must be string type")
        plt.title(title)

    @classmethod
    def set_eda_folder(cls, folder:str) -> None:
        """
        Set the folder path for saving figures.

        Args:
            folder (str): Folder path for saving figures.

        Raises:
            TypeError: If argument folder is not type string.
            ValueError: If argument folder is not a valid path.
        """
        if not isinstance(folder, str):
            raise TypeError("Parameter folder must be string type.")
        if not os.path.isdir(folder) and folder!='':
            raise ValueError("Parameter folder is not a valid path.")
        if os.path.isdir(folder) and not folder.endswith('/'):
            folder = folder + '/'
        cls.fig_folder = f"{folder}"

    @classmethod
    def table_fig(cls,
        dataframe:pd.DataFrame,
        **kwargs
    ) -> None:
        """
        Create and display/save a table figure from a DataFrame.

        ---------------------------------------------------------

        If receive a name for fig_name it will save the figure
        in the selected folder. If not, it will display the figure.

        ----------------------------------------------------------

        Args:
            dataframe (pd.DataFrame): Input data.
            fig_name (str, optional): Name of the figure for saving. Defaults to None.
            title (str, optional): Title of the figure. Defaults to None.

        Raises:
            TypeError: If argument dataframe is not a DataFrame.
            ValueError: If the DataFrame has rows or columns greater than 40

        Usage:
        ```
        data = pd.read_csv("data.csv")
        # If you want to save figure
        Figures.table_fig(dataframe = data.head(), fig_name = 'name', title = 'title')
        # If you want to display figure
        Figures.table_fig(dataframe = data.head(), title = 'title')
        ```
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Argument dataframe must be a Pandas Dataframe")
        if any(i > 40 for i in dataframe.shape):
            raise ValueError("Dataframe is too large to generate a figure")

        rows, columns = dataframe.shape
        plt.figure(figsize=(columns*2, rows*0.33 if rows>6 else 2))
        plt.axis('off')
        plt.table(
            cellText=dataframe.values,
            colLabels=dataframe.columns,
            loc='center',
            cellLoc='center'
        )
        plt.tight_layout()
        cls.__finalize_fig(**kwargs)

    @classmethod
    def histogram_fig(cls,
        series:pd.Series,
        fig_name: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Create and display/save a histogram figure from a Series.

        ---------------------------------------------------------

        If receive a name for fig_name it will save the figure
        in the selected folder. If not, it will display the figure

        ----------------------------------------------------------

        Args:
            series (pd.Series): Input data.
            fig_name (str, optional): Name of the figure for saving. Defaults to None.
            title (str, optional): Title of the figure. Defaults to None.
            stat (str, optional): If declared, must be 'count' or 'density'. Defaults to 'count'.
            kde (bool, optional): Defaults to False.

        Raises:
            TypeError: If argument series is not pandas series type.
            TypeError: If unexpected argument.
            ValueError: If parameter stat is not either 'count' or 'density'
            TypeError: If parameter kde is not boolean type

        Usage:
        ```
        data = pd.read_csv("data.csv")
        # If you want to save figure
        Figures.histogram_fig(
                series = data.column_name,
                fig_name = "histogram",
                title = "My Histogram",
                stat = 'density',
                kde = True
            )
        # If you want to display figure
        Figures.histogram_figure(
                series = data.column_name,
                title = "My Histogram",
                stat = 'density',
                kde = True
            )
        ```
        """
        if not isinstance(series, pd.Series):
            raise TypeError("Argument series must be type pandas series.")
        if set(kwargs) - set(['stat', 'kde']):
            raise TypeError("Got an unexpected keyword argument.")
        if kwargs.get('stat') and kwargs.get('stat') not in ['density', 'count']:
            raise ValueError("If declared, 'stat' parameter must be 'count' or 'density'")
        if kwargs.get('kde') and not isinstance(kwargs.get('kde'), bool):
            raise TypeError("If declared, 'kde' parameter must be type bool")
        sns.histplot(series, **kwargs)
        plt.tight_layout()
        cls.__finalize_fig(fig_name, title)

    @classmethod
    def heatmap_fig(cls,
        dataframe:pd.DataFrame,
        **kwargs
    ) -> None:
        """
        Create and display/save a heatmap figure from a DataFrame.

        ---------------------------------------------------------

        If receive a name for fig_name it will save the figure
        in the selected folder. If not, it will display the figure

        ----------------------------------------------------------

        Args:
            dataframe (pd.DataFrame): Input data.
            fig_name (str, optional): Name of the figure for saving. Defaults to None.
            title (str, optional): Title of the figure. Defaults to None.

        Raises:
            TypeError: If argument dataframe is not pandas dataframe type.

        Usage:
        ```
        data = pd.read_csv("data.csv")
        # If you want to save figure
        Figures.heatmap_fig(
                dataframe = data,
                fig_name = "heatmap",
                title = "My heatmap"
            )
        # If you want to display figure
        Figures.heatmap_fig(
                dataframe = data,
                title = "My heatmap"
            )
        ```

        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Argument dataframe must be pandas dataframe type")
        plt.figure(figsize=(20, 10))
        sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.tight_layout()
        cls.__finalize_fig(**kwargs)

    @classmethod
    def roc_curve_fig(cls,
        models:list,
        x_test:pd.DataFrame,
        y_test:pd.Series,
        **kwargs) -> None:
        """
        Create and display/save a ROC figure from a list of models,
        based on test data.
        """

        plt.figure(figsize=(15, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            display = plot_roc_curve(models[0], x_test, y_test)
            for model in models[1:]:
                plot_roc_curve(model, x_test, y_test, ax=display.ax_)

        plt.legend(loc='lower right')
        plt.tight_layout()
        cls.__finalize_fig(**kwargs)

    @classmethod
    def classification_report_fig(cls,
        y_test_data:pd.Series,
        y_test_pred:np.ndarray,
        y_train_data:pd.Series,
        y_train_pred:np.ndarray,
        **kwargs) -> None:
        """
        Create and display/save a classification report figure
        based on train and test data
        """

        plt.rc('figure', figsize=(5, 6))

        # Add Title to test figure
        plt.text(
            x = 0.01,
            y = 0.85,
            s = str(f"{kwargs.get('title') if kwargs.get('title') else ''} Test"),
            fontdict = {'fontsize': 10},
            fontproperties='monospace'
        )
        # Add test figure text
        plt.text(
            x = 0.01,
            y = 0.5,
            s = str(classification_report(y_test_data, y_test_pred)),
            fontdict = {'fontsize': 10},
            fontproperties = 'monospace'
        )
        # Add Title to train figure
        plt.text(
            x = 0.01,
            y = 0.45,
            s = str(f"{kwargs.get('title') if kwargs.get('title') else ''} Train"),
            fontdict = {'fontsize': 10},
            fontproperties = 'monospace'
        )
        # Add train figure text
        plt.text(
            x = 0.01,
            y = 0.1,
            s = str(classification_report(y_train_data, y_train_pred)),
            fontdict = {'fontsize': 10},
            fontproperties = 'monospace'
        )

        plt.axis('off')
        plt.tight_layout()
        cls.__finalize_fig(**kwargs)

    @classmethod
    def bar_fig(cls,
        values:np.ndarray,
        names:List[str],
        **kwargs
    ) -> None:
        """
        Create and display/save a bar figure based on alist of names
        and a array of values.
        """
        plt.figure(figsize=(20,5))
        plt.bar(range(values.shape[0]), values)
        plt.xticks(range(values.shape[0]), names, rotation=90)
        plt.tight_layout()
        cls.__finalize_fig(**kwargs)

    @classmethod
    def tree_explainer_fig(cls,
        model:RandomForestClassifier,
        x_test:pd.DataFrame,
        **kwargs
    ) -> None:
        """
        Create and display/save a explainer for a random forest model.
        """
        plt.subplots()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
        cls.__finalize_fig(**kwargs)


if __name__ == '__main__':
    # Load dataframe
    df = pd.DataFrame(data = {'column 1':[1,2,3], 'column 2':[1,2,3]})
    # This method will display data head
    Figures.table_fig(dataframe = df.head())
    # This method wil display a heatmap from dataframe
    Figures.heatmap_fig(dataframe=df)
    # This method wil display a classification report
    Figures.classification_report_fig(
        y_test_data = pd.Series([1,2,3]),
        y_test_pred = np.array([1,2,4]),
        y_train_data = pd.Series([6,8,9,10,11]),
        y_train_pred = np.array([6,8,9,10,11])
    )
