import os
import numpy as np
import pandas as pd

import utils, paths
from sklearn.model_selection import cross_validate


def get_cross_val_results(model_type, clf, X, y):
    utils.validate_model_type_name(model_type=model_type)
    if model_type == "classification":

        scoring = ["accuracy", "balanced_accuracy", "f1_micro", "f1_macro", "precision_micro",
                   "recall_micro", "precision_macro", "recall_macro"]
    else:
        scoring = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                   "neg_mean_squared_log_error", "neg_median_absolute_error"]
    scores = cross_validate(clf, X=X, y=y, scoring=scoring, cv=10)
    result = pd.DataFrame.from_dict(scores, orient='columns')
    return result


def generate_person_specific_results(dataset, signal, model_type):
    data = utils.get_combined_data(dataset=dataset, signal=signal)
    subject_id_col = utils.get_subject_id_column(dataset)
    subjects_ids = sorted(data[subject_id_col].unique())

    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)
    models = utils.get_model(model_type)
    for clf in models:
        model_name = type(clf).__name__
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "person-specific",
                                                             model_name))
        clf = utils.get_pipeline_model(clf)
        for subject_id in subjects_ids:
            df = data.loc[data[subject_id_col] == subject_id]
            X = df[features]
            y = df[target]
            result = get_cross_val_results(model_type, clf, X, y)
            print("user_id {0}".format(subject_id))
            result.index.name = 'CV Fold'
            result.to_csv(os.path.join(out_dir, "subject_" + str(int(subject_id)) + ".csv"), index=True)


if __name__ == "__main__":
    datasets = ["swell", "wesad"]
    signals = ["hrv", "eda"]
    types = ["classification", "regression"]
    for dataset in datasets:
        for signal in signals:
            for model_type in types:
                generate_person_specific_results(dataset=dataset, signal=signal, model_type=model_type)
