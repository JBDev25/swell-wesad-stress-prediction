from sklearn import metrics
import pandas as pd
import numpy as np


def extract_dataframe(signal,dataset):
    df_generic = pd.read_csv(f'../dataset/{signal}/{dataset}/generic.csv')
    df_callibration = pd.read_csv(f'../dataset/{signal}/{dataset}/calibration.csv')
    df_test = pd.read_csv(f'../dataset/{signal}/{dataset}/test.csv')
    return df_generic,df_callibration,df_test


def get_classification_metrics(predictions, y_test):
    accuracy = metrics.accuracy_score(y_pred=predictions, y_true=y_test)
    precision = metrics.precision_score(y_pred=predictions, y_true=y_test, average='weighted')
    f1_score = metrics.f1_score(y_pred=predictions, y_true=y_test, average='weighted')
    recall = metrics.recall_score(y_pred=predictions, y_true=y_test, average='weighted')
    result = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
              "F1 score": f1_score}
    return pd.DataFrame(result, index=[0]).transpose()


def get_regression_metrics(predictions, y_test):
    mae = metrics.mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    return pd.DataFrame({'MAE': mae, 'RMSE': rmse}, index=[0]).transpose()

def get_target_var(dataset,model_type):
    if dataset == 'wesad' and model_type =='reg':
        return 'SSSQ'
    elif dataset == 'swell' and model_type == 'reg':
        return 'NasaTLX'
    else:
        return 'condition'
    
    
def get_calibration_subjects(signal,dataset):
    if dataset == 'wesad':
        return [2, 6, 9, 11]
    else:
        return [1, 2, 5, 25]
    


