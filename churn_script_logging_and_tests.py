"""
 Author : Shariq
 Purpose : Unitest for churn library
 Date : 27 June 202
"""
import os
import logging
import pytest
import churn_library as cls
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

PATH = "data/bank_data.csv"


@pytest.fixture(scope="module")
def read_df():
    """
     reads the data
     input : Nothing
     output : returns the dataframe
    """
    return cls.import_data(PATH)


@pytest.fixture(scope="module")
def feature_eng_analysis(read_df):
    """
      Performs the feature engineering and returns X_train,X_test,y_train,y_test
      input  : dataframe
      output : X_train,X_test,y_train,y_test
    """
    return cls.perform_feature_engineering(read_df)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    input : dataframe
    output : None
    '''
    try:
        df = cls.import_data(PATH)
        logging.info("Imported the data successfully")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(read_df):
    '''
    test perform eda function
    input : dataframe
    output : None
    '''
    col_names = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct"]
    cls.perform_eda(read_df)
    try:
        for val in col_names:
            assert "Distribution_of_" + val + \
                ".png" in os.listdir("images/eda/")
    except AssertionError as err:
        logging.error("ERROR: File not found")
        raise err
    logging.info("EDA succesfuly performed")


def test_encoder_helper(read_df):
    '''
    test encoder helper
    input : dataframe
    output : None
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df = cls.encoder_helper(read_df, cat_columns)
    try:
        for col in cat_columns:
            assert col + "_Churn" in df.columns
    except AssertionError as err:
        logging.error("ERROR:  Encoding not performed properly")
        raise err
    logging.info("Encoder_helper successfully performed")


def test_perform_feature_engineering(read_df):
    '''
    test perform_feature_engineering
    input  : the feature_eng_analysis output consisting of X_train, X_test, y_train, y_test
    output : None
    '''
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(read_df)

    try:
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
    except AssertionError as err:
        logging.error("ERROR:  Feature Engineering not performed properly")
        raise err
    logging.info("Feature Engineering  successfully performed")
    return X_train, X_test, y_train, y_test


def test_train_models(feature_eng_analysis):
    '''
    test train_models
    input  : the feature_eng_analysis output consisting of X_train, X_test, y_train, y_test
    output : None
    '''
    X_train, X_test, y_train, y_test = feature_eng_analysis
    cls.train_models(X_train, X_test, y_train, y_test)
    try:
        lis = ["logistic_model.pkl", "rfc_model.pkl"]
        for file in os.listdir("models"):
            assert file in lis
        lis_plot = ["ROC_plot.png", "Feature_Importance.png"]
        for file in lis_plot:
            assert file in os.listdir("images/results/")

    except AssertionError as err:
        logging.error("ERROR:  Training model not performed properly")
        raise err
    logging.info("Training model successfully performed")


if __name__ == "__main__":
    artifacts_dir = ["images/eda", "images/results", "models"]
    for directory in artifacts_dir:
        for files in os.listdir(directory):
            if(os.path.exists(os.path.join(directory, files))):
                os.remove(os.path.join(directory, files))

    df = test_import()
    test_eda(df)
    test_encoder_helper(df)
    val = test_perform_feature_engineering(df)
    test_train_models(val)
