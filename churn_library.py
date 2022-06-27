"""
Author : Shariq Ahmad
Purpose : Predict the churn of the customer
Date : 26 June 2022

"""
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    #pp = PdfPages('images/eda/figure.pdf')

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    col_names = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct"]
    for col in col_names:
        plt.figure(figsize=(20, 10))
        if col == "Churn":
            df[col].hist()
        elif col == "Customer_Age":
            df[col].hist()
        elif col == "Marital_Status":
            df[col].value_counts('normalize').plot(kind='bar')
        elif col == "Total_Trans_Ct":
            sns.histplot(df[col], stat='density', kde=True)
        plt.title(f"Distribution of {col}", fontsize=24,)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(os.path.join("images/eda/", f"Distribution_of_{col}.png"))

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join("images/eda/","Heatmap_distribution.png"))
    # pp.savefig(fig5)
    # pp.close()


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        gender_groups = df.groupby(cat).mean()['Churn']
        # Map function will actually replace the loop requirment in orignal
        # code
        df[cat + "_Churn"] = df[cat].map(gender_groups)

    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(df, cat_columns)

    y = df['Churn']
    keep_cols = [
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

    X = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


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

    plt.figure()
    plt.rc('figure', figsize=(10, 10))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/RF_Metrices_result.png')

    plt.figure()
    plt.rc('figure', figsize=(10, 10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/LR_Metrices_result.png')


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
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 20))
    # Create plot title
    plt.title("Feature Importance", fontsize=20)
    plt.ylabel('Importance', fontsize=20)
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90, fontsize=10)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(output_pth,'Feature_Importance.png'))


def train_models(X_train, X_test, y_train, y_test):
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

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [100],
        'criterion': ['entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    #print(cv_rfc.best_estimator_)
    lrc.fit(X_train, y_train)
     
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

def prediction(X_train, X_test, y_train, y_test):

    print("Inside_plotting_check")
    model_rf = joblib.load('./models/rfc_model.pkl')
    model_lr = joblib.load('./models/logistic_model.pkl')

    y_train_preds_rf = model_rf.predict(X_train)
    y_test_preds_rf = model_rf.predict(X_test)
    y_train_preds_lr = model_lr.predict(X_train)
    y_test_preds_lr = model_lr.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(model_lr, X_test, y_test)
    plot_roc_curve(model_rf, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('images/results/ROC_plot.png')

    feature_importance_plot(model_rf, X_test, "images/results/")


if __name__ == "__main__":
    PATH = ("data/bank_data.csv")
    data = import_data(PATH)
    perform_eda(data)

    X_train, X_test, y_train, y_test = perform_feature_engineering(data)
    train_models(X_train, X_test, y_train, y_test)
    prediction(X_train, X_test, y_train, y_test)
