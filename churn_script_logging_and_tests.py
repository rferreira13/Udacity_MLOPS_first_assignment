"""
Module to perform tests on script churn_library.py

This module may be run interactively with ipython or through pytest

Author: Rafael Ferreira
Date: September 2023
"""
import os
import logging
import pytest
import pandas as pd
import churn_library as cls
import helpers.params as p

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename=p.LOG_FILE_PATH,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="module", params=p.IMPORT_DATA_TESTS)
def filepath_test(request):
    """Fixture for test_import function"""
    return request.param

# pylint: disable=redefined-outer-name
def test_import(filepath_test):
    """Function to test import_data"""
    logging.info("LOG for path -- %s --", filepath_test)
    try:
        cls.import_data(path = filepath_test)
        logging.info("SUCCESS")
    except FileNotFoundError as exc:
        logging.error("FileNotFoundError with message -- %s --", exc)
    except pd.errors.EmptyDataError as exc:
        logging.error("EmptyDataError with message -- %s --", exc)
    except pd.errors.ParserError as exc:
        logging.error("ParserError with message -- %s --", exc)
    except ValueError as exc:
        logging.error("ValueError with message -- %s --", exc)

@pytest.fixture(scope="module", params=p.DATA_TESTS)
def data_test(request):
    """Fixture for test_import function"""
    return request.param

@pytest.fixture(scope="module", params=p.EDA_OUTPUT_PTH_TESTS)
def eda_output_pth_test(request):
    """Fixture for test_import function"""
    return request.param

def test_eda(data_test, eda_output_pth_test):
    """
    test encoder helper
    """
    logging.info(
        "LOG for eda_data -- %s --, and eda_output_pth -- %s --",
        data_test, eda_output_pth_test
    )
    try:
        cls.perform_eda(eda_data=data_test,
            eda_output_pth=eda_output_pth_test)
        logging.info(
            "SUCCESS"
        )
    except TypeError as exc:
        logging.error(
            "TypeError with message -- %s --", exc
        )
    except ValueError as exc:
        logging.error(
            "ValueError with message -- %s --", exc
        )

@pytest.fixture(scope="module", params=p.CATEGORY_LST_TESTS)
def category_lst_test(request):
    """Fixture for test test_encoder_helper function"""
    return request.param

@pytest.fixture(scope="module", params=p.RESPONSE_TESTS)
def response_test(request):
    """Fixture for test test_encoder_helper function"""
    return request.param

def test_encoder_helper(category_lst_test, response_test):
    """
    test encoder helper
    """
    logging.info(
        "LOG for category_lst -- %s --, and response -- %s --",
        category_lst_test, response_test
    )
    try:
        data = cls.import_data(p.BANK_DATA_PTH)
        feat_eng_obj = cls.FeatureEngineering(data)
        dependent_data = feat_eng_obj.select_response(
            column = p.DEPENDENT_COLUMN,
            category = p.DEPENDENT_CATEGORY
        )
        feat_eng_obj.remove_columns(columns = [p.DEPENDENT_COLUMN,])
        feat_eng_obj.add_column(
            column_data = dependent_data,
            column_name = p.RESPONSE
        )
        feat_eng_obj.encode(category_lst = category_lst_test, response = response_test)
        logging.info(
            "SUCCESS"
        )
    except KeyError as exc:
        logging.error(
            "KeyError with message -- %s --", exc
        )

@pytest.fixture(scope="module", params=p.SIZE_TESTS)
def size_test(request):
    """Fixture for test test_encoder_helper function"""
    return request.param

@pytest.fixture(scope="module", params=p.DEPENDENT_COLUMN_TESTS)
def dependent_column_test(request):
    """Fixture for test test_encoder_helper function"""
    return request.param

@pytest.fixture(scope="module", params=p.DEPENDENT_CATEGORY_TESTS)
def dependent_category_test(request):
    """Fixture for test test_encoder_helper function"""
    return request.param

def test_perform_feature_engineering(
        data_test,
        size_test,
        dependent_column_test,
        dependent_category_test,
        response_test
):
    """
    test encoder helper
    """
    logging.info(
        "LOG for feat_data -- %s --, and size -- %s --,"
        "and dependent_column -- %s --, and dependent_category -- %s --,"
        "and response -- %s --",
        data_test,
        size_test,
        dependent_column_test,
        dependent_category_test,
        response_test
    )
    try:
        data = cls.import_data(p.BANK_DATA_PTH)
        feat_eng_obj = cls.FeatureEngineering(data)
        dependent_data = feat_eng_obj.select_response(
            column = p.DEPENDENT_COLUMN,
            category = p.DEPENDENT_CATEGORY
        )
        feat_eng_obj.remove_columns(columns = [p.DEPENDENT_COLUMN,])
        feat_eng_obj.add_column(
            column_data = dependent_data,
            column_name = p.RESPONSE
        )
        feat_eng_obj.encode(category_lst = category_lst_test, response = response_test)
        logging.info(
            "SUCCESS"
        )
    except TypeError as exc:
        logging.error(
            "TypeError with message -- %s --", exc
        )
    except ValueError as exc:
        logging.error(
            "ValueError with message -- %s --", exc
        )
    except KeyError as exc:
        logging.error(
            "KeyError with message -- %s --", exc
        )

if __name__ == '__main__':

    # Tests for function import_data
    for path in p.IMPORT_DATA_TESTS:
        test_import(path)

    # Tests for function encode
    for eda_data in p.DATA_TESTS:
        for eda_output_pth in p.EDA_OUTPUT_PTH_TESTS:
            test_eda(
                data_test = eda_data,
                eda_output_pth_test = eda_output_pth
            )

    # Tests for function encode
    for category_lst in p.CATEGORY_LST_TESTS:
        for response in p.RESPONSE_TESTS:
            test_encoder_helper(
                category_lst_test = category_lst,
                response_test = response
            )

    # Tests for function encode
    for feat_data in p.DATA_TESTS:
        for size in p.SIZE_TESTS:
            for dependent_column in p.DEPENDENT_COLUMN_TESTS:
                for dependent_category in p.DEPENDENT_CATEGORY_TESTS:
                    for response in p.RESPONSE_TESTS:
                        test_perform_feature_engineering(
                            data_test=feat_data,
                            size_test=size,
                            dependent_column_test=dependent_column,
                            dependent_category_test=dependent_category,
                            response_test=response
                        )
