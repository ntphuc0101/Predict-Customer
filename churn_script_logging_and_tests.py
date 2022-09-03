import os
import logging
import churn_library as cls
import seaborn as sns

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
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


def test_eda_output(path, input_image):
    try:
        assert os.path.isfile('{0}/{1}'.format(path, input_image)) is True
        logging.info('File %s was found', input_image)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err


def test_eda(perform_eda, import_data):
    '''
    test perform eda function, check input and output
    '''
    path = './images/eda/'
    try:
        data = import_data(r"./data/bank_data.csv")
        print(data)
        perform_eda(data)
        logging.info("Testing perform_eda: SUCCESS")
    except AttributeError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err
    # Assert if `Churn.png` is created
    input_image = 'Churn.png'
    test_eda_output(path, input_image)
    # Assert if `Customer_Age.png` is created
    input_image = 'Customer_Age.png'
    test_eda_output(path, input_image)
    # Assert if `Marital_Status.png` is created
    input_image = 'Marital_Status.png'
    test_eda_output(path, input_image)
    # Assert if `heatmap.png` is created
    input_image = 'heatmap.png'
    test_eda_output(path, input_image)


def test_encoder_helper(encoder_helper, import_data):
    '''
    test encoder helper: check input of encoder whether is correct type or categories
    '''
    df = import_data("./data/bank_data.csv")
    try:
        input_column = [
            'Gender', 'Education_Level',
            'Marital_Status', 'Income_Category',
            'Card_Category'
        ]
        encoded_df = encoder_helper(df=df, category_lst=[], response='Churn')
        logging.info("Testing encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing encoder_helper: There are column names that doesn't exist in your dataframe")
        raise err
    try:
        encoded_df = encoder_helper(
            df=df, category_lst=input_column, response='Churn')
        # Name of columns should be the different
        assert encoded_df.columns.equals(df.columns) is False
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=[]): SUCCESS")
        # Data should be different
        assert encoded_df.equals(df) is False
        # Number of columns in encoded_df is the sum of columns in data_frame
        # and the newly created columns from cat_columns
        assert len(encoded_df.columns) == len(df.columns) + len(input_column)
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): SUCCESS")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    # Load the DataFrame
    df = cls.import_data("./data/bank_data.csv")
    try:
        (_, X_test, _, _) = perform_feature_engineering(
            df=df,
            response='Churn')
        assert 'Churn' in df.columns
        # `Test` must be present in `df`
        logging.info(
            "Testing perform_feature_engineering: `'Churn'` column is present: SUCCESS")
    except KeyError as err:
        logging.error(
            'The `Test` column is not present in the DataFrame: ERROR')
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    # Load the DataFrame
    df = cls.import_data("./data/bank_data.csv")
    # Feature engineering
    (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
        df=df,
        response='Churn')
    train_models(X_train, X_test, y_train, y_test)
    logging.info("Testing test_train_models(train_models)")
    path = './models/'
    input_models = ['rfc_model.pkl', 'logistic_model.pkl']
    for input_model in input_models:
        test_eda_output(path, input_model)
    path = './images/results/'
    input_images = [
        'roc_curve_result.png',
        'rf_results.png',
        'logistic_results.png',
        'feature_importances.png']
    for input_image in input_images:
        test_eda_output(path, input_image)


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda, cls.import_data)
    test_encoder_helper(cls.encoder_helper, cls.import_data)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
