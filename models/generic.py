import os
from sklearn.utils import shuffle
import utils, paths
import pandas as pd


def generate_generic_results(dataset, signal, model_type):
    data = utils.get_combined_data(dataset=dataset, signal=signal)
    subject_id_col = utils.get_subject_id_column(dataset)
    subjects_ids = sorted(data[subject_id_col].unique())
    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)
    clf = utils.get_base_classifier()
    model_name = type(clf).__name__
    out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "generic-model",
                                                            model_name))
    for subject_id in subjects_ids:
        clf = utils.get_base_classifier()
        train_subjects = [x for x in subjects_ids if x != subject_id]
        test = data.loc[data[subject_id_col] == subject_id]
        train = data.loc[data[subject_id_col].isin(train_subjects)]
        train = shuffle(train)
        test = shuffle(test)
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]
        clf.fit(X_train, y_train)
        # Get predictions
        predictions = clf.predict(X_test)
        result = utils.get_prediction_metrics(model_type, predictions=predictions, y_test=y_test)
        print("user_id {0} \t{1}".format(subject_id, result.transpose()))

        result.to_csv(os.path.join(out_dir, "subject_" + str(int(subject_id)) + ".csv"), index=True)


def generate_combined_general_results(dataset, signal, model_type):
    data = utils.get_combined_data(dataset=dataset, signal=signal)
    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)
    models = utils.get_model(model_type)
    for clf in models:
        model_name = type(clf).__name__
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "combined-general-model",
                                                             model_name))
        clf = utils.get_pipeline_model(clf)
        X = data[features]
        y = data[target]
        result = get_cross_val_results(model_type, clf, X, y)
        result.index.name = 'CV Fold'
        result.to_csv(os.path.join(out_dir, model_name + ".csv"), index=True)


def get_cross_val_results(model_type, clf, X, y):
    from sklearn.model_selection import cross_validate
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
