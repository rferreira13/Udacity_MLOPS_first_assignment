"""
Module to configure log file for pytest

Author: Rafael Ferreira
Date: September 2023
"""
import logging
import os

def pytest_configure():
    """
    Function to configure log file for pytest
    """
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    logging.basicConfig(
        filename='./logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
