import numpy as np
from sklearn import metrics
import os

import pandas as pd
import paths
from lightgbm import LGBMClassifier


def get_regression_metrics(predictions, y_test):
    mae = metrics.mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    return pd.DataFrame({'MAE': mae, 'RMSE': rmse}, index=[0]).transpose()


def get_classification_metrics(predictions, y_test):
    accuracy = metrics.accuracy_score(y_pred=predictions, y_true=y_test)
    precision = metrics.precision_score(y_pred=predictions, y_true=y_test, average='weighted')
    f1_score = metrics.f1_score(y_pred=predictions, y_true=y_test, average='weighted')
    recall = metrics.recall_score(y_pred=predictions, y_true=y_test, average='weighted')
    result = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
              "F1 score": f1_score}
    return pd.DataFrame(result, index=[0]).transpose()


def get_important_features(dataset, signal, model_type):
    in_file = os.path.join(paths.result_directory(), "feature-ranks", signal,
                           model_type, dataset, "features-ranks.csv")
    feature_count = 0
    if dataset == "swell" and signal == "eda":
        feature_count = 46
    if dataset == "swell" and signal == "hrv":
        feature_count = 75
    if dataset == "wesad" and signal == "hrv":
        feature_count = 40
    if dataset == "wesad" and signal == "eda":
        feature_count = 45
    data = pd.read_csv(in_file)
    features = data[list(data)[0]].head(feature_count).tolist()
    return features


def get_base_classifier():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    # define baseline model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(64, input_dim=94, activation='relu'))
        model.add(Dense(32, input_dim=64, activation='relu'))
        model.add(Dense(16, input_dim=32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
 
    base_estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=100, verbose=1)
    return base_estimator


def get_base_regressor():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    # define baseline model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(64, input_dim=94, activation='relu'))
        model.add(Dense(32, input_dim=64, activation='relu'))
        model.add(Dense(16, input_dim=32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        return model
 
    base_estimator = KerasRegressor(build_fn=baseline_model, epochs=5, batch_size=50, verbose=1)
    return base_estimator


def get_subjects_ids(dataset):
    validate_dataset_name(dataset)
    if dataset == "swell":
        return [x for x in range(1, 26) if x not in [8, 11, 14, 15, 23]]
    else:
        return [x for x in range(2, 18) if x not in [5, 12]]


def get_prediction_metrics(model_type, predictions, y_test):
    validate_model_type_name(model_type=model_type)
    if model_type == "classification":
        return get_classification_metrics(predictions=predictions, y_test=y_test)
    else:
        return get_regression_metrics(predictions=predictions, y_test=y_test)


def get_prediction_target(dataset, model_type):
    validate_dataset_name(dataset)
    validate_model_type_name(model_type)
    target = None
    if model_type == "classification":
        target = 'condition'
    elif dataset == "swell" and model_type == "regression":
        target = 'NasaTLX'
    elif dataset == "wesad" and model_type == "regression":
        target = "SSSQ"
    return target


def get_combined_data(dataset, signal):
    in_file = os.path.join(paths.data_directory(), signal, dataset, "all-samples.csv")
    return pd.read_csv(in_file)


def get_model(model_type):
    if model_type == "classification":
        clf = get_classifier()
    else:
        clf = get_regressor()
    return clf


def validate_model_type_name(model_type):
    if model_type not in ["classification", "regression"]:
        raise ValueError("{0} is an invalid model type".format(model_type))


def validate_dataset_name(dataset):
    if dataset not in ["wesad", "swell"]:
        raise ValueError("{0} is an invalid dataset name".format(dataset))


def validate_signal_name(signal):
    if signal not in ["eda", "hrv"]:
        raise ValueError("{0} is an invalid dataset name".format(signal))


if __name__ == '__main__':
    f1 = get_important_features(dataset="swell", signal="eda", model_type="regression")
    f2 = get_important_features(dataset="swell", signal="eda", model_type="classification")
    f3 = get_important_features(dataset="swell", signal="hrv", model_type="regression")
    f4 = get_important_features(dataset="swell", signal="hrv", model_type="classification")
    f5 = get_important_features(dataset="wesad", signal="hrv", model_type="classification")
    f6 = get_important_features(dataset="wesad", signal="hrv", model_type="regression")
    f7 = get_important_features(dataset="wesad", signal="eda", model_type="regression")
    f8 = get_important_features(dataset="wesad", signal="eda", model_type="classification")


def get_subject_id_column(dataset):
    if dataset == "swell":
        subject_id_col = "subject_id"
    else:
        subject_id_col = "subject id"
    return subject_id_col


def get_calibration_sample_sizes(dataset):
    validate_dataset_name(dataset=dataset)
    if dataset == "swell":
        return np.arange(0, 401, 40)
    else:
        return np.arange(0, 401, 20)
