'''
In this file, there are functions of churn customer analysis
Author: Phuc Nguyen Thai Vinh
Date: 03-Sept-2022
'''

# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_roc_curve

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import joblib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    print("[INFO] read the data from {0}".format(pth))
    data = pd.read_csv(pth)
    # Flag churn customer
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    # Drop column, first column no values (just index), CLIENTNUM contains no
    # useful inforamtion , for attrition_Flag, we replace the value with 0 and
    # 1
    data_frame = data.drop(
        ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], axis=1)
    return data_frame


def plot_historgram(df, num_columns, path):
    for num_col_col in num_columns.columns:
        plt.figure(figsize=(20, 10))
        df[num_col_col].hist()
        print("[INFO] Create Histogram plot of {0} column".format(num_col_col))
        plt.title("{0} Distribution".format(num_col_col))
        plt.savefig('{0}/{1}.png'.format(path, num_col_col))
        plt.close()


def plot_bar_chart(df, cat_columns, path):
    for cat_col_name in cat_columns.columns:
        plt.figure(figsize=(20, 10))
        df[cat_col_name].value_counts('normalize').plot(kind='bar')
        print("[INFO] Create Bar plot of {0} column".format(cat_col_name))
        plt.title("{0} Distribution".format(cat_col_name))
        plt.savefig('{0}/{1}.png'.format(path, cat_col_name))
        plt.close()


def plot_heatmap(df, path):
    sns.set()
    print("[INFO] Create Heatmap Plot")
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('{0}/heatmap.png'.format(path))
    plt.close()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    print('======================================================')
    print("[INFO] Performing EDA")
    num_columns = df.select_dtypes(include="number")
    cat_columns = df.select_dtypes(exclude="number")
    path = './images/eda'
    # Ploting histogram
    plot_historgram(df, num_columns, path)
    # Ploting bar chart
    plot_bar_chart(df, cat_columns, path)
    # heatmap plot
    plot_heatmap(df, path)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Copy DataFrmae
    encoder_df = df.copy(deep=True)

    for category in category_lst:
        column_lst = []
        column_groups = df.groupby(category).mean()['Churn']
        column_groups_tmp = df.groupby(category).mean()

        for category_name in df[category]:
            column_lst.append(column_groups.loc[category_name])

        if response:
            encoder_df[category + '_' + response] = column_lst
        else:
            encoder_df[category] = column_lst

    return encoder_df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    print("[INFO] Feature engineering")
    # categorical features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    # feature engineering
    encoded_df = encoder_helper(
        df=df,
        category_lst=cat_columns,
        response=response)

    # target feature
    label = encoded_df['Churn']

    # Create dataframe
    Input = pd.DataFrame()

    cols_remain = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # Features DataFrame
    Input[cols_remain] = encoded_df[cols_remain]

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        Input, label, test_size=0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def my_classification_report(y_train,
                             y_test, y_train_preds,
                             y_test_preds, path,
                             fig_name, name):
    '''
    This function is used to write Random forest report
    '''
    # Random forest report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str(name),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str(name),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('{0}/{1}'.format(path, fig_name))
    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    print("[INFO] Compute classification Report of images")

    # Random forest report
    path = './images/results/'
    fig_name = 'rf_results.png'
    name = 'Random Forest Train'
    my_classification_report(y_train,
                             y_test, y_train_preds_rf,
                             y_test_preds_rf, path,
                             fig_name, name)

    # Logistic regression report
    path = './images/results/'
    fig_name = 'logistic_results.png'
    name = 'Logistic Regression Train'
    my_classification_report(y_train,
                             y_test, y_train_preds_lr,
                             y_test_preds_lr, path,
                             fig_name, name)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(25, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')


def save_model(cv_rfc, lrc):
    '''
    This function is used to save best model
    cv_rfc, lrc are two models for random forests and logistic regression
    '''
    path = './models'
    print("[INFO] Save the model objects")
    joblib.dump(cv_rfc.best_estimator_, path + '/rfc_model.pkl')
    joblib.dump(lrc, path + '/logistic_model.pkl')


def test_model(cv_rfc, lrc, x_train, x_test):
    '''
    This function is used to save best model
    cv_rfc, lrc are two models for random forests and logistic regression
    '''
    # Choosing best estimator
    print("[INFO] Test the model objects")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr  = lrc.predict(x_test)

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


def plot_ROC_curve(lrc, cv_rfc, X_test, y_test):
    '''
    This function is used to plot ROC curve
    X_test: input for testing
    y_test: output for testing
    '''
    print("[INFO] Plotting the model objects")
    result_path = './images/results/'
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_,
                              X_test,
                              y_test,
                              ax=axis,
                              alpha=0.8)
    fname = result_path + '/roc_curve_result.png'
    plt.savefig(fname=fname)


def train_models(X_train,
                 X_test,
                 y_train,
                 y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=1000)

    # Parameters for Grid Search
    param_grid = {'n_estimators': [200, 500],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [4, 5, 100],
                  'criterion': ['gini', 'entropy']}

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    print("[INFO] Training the model objects")
    cv_rfc.fit(X_train, y_train)

    # LogisticRegression
    lrc.fit(X_train, y_train)

    # Save models
    save_model(cv_rfc, lrc)

    # Test models
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = test_model(
        cv_rfc, lrc, X_train, X_test)
    # Plot ROC curve
    plot_ROC_curve(lrc, cv_rfc, X_test, y_test)
    
    # Compute and results:
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            X_data=X_test,
                            output_pth='./images/results/')


if __name__ == "__main__":
    file = import_data(r"./data/bank_data.csv")

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        df=file, response='Churn')

    # Model training,prediction and evaluation
    train_models(
        X_train=X_TRAIN,
        X_test=X_TEST,
        y_train=Y_TRAIN,
        y_test=Y_TEST)
