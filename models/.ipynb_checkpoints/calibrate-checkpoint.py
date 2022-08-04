import os
import pandas as pd
import paths, utils
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def calibrate_model(model, generic_samples, calibration_samples, target, features):
    from sklearn.base import clone
    train = pd.concat([generic_samples, calibration_samples])
    train = shuffle(train)
    X_train = train[features]
    y_train = train[target]
    # This is not actually required... but I am "reseting"
    # the modal just to make sure it is not using the old mode
    clf = clone(model)
    clf.fit(X_train, y_train)
    return clf
def get_data(dataset, signal):
    root_dir = os.path.join(paths.data_directory(), signal, dataset)
    calibration = pd.read_csv(os.path.join(root_dir, "calibration.csv"))
    generic = pd.read_csv(os.path.join(root_dir, "generic.csv"))
    test = pd.read_csv(os.path.join(root_dir, "test.csv"))
    return generic, calibration, test


def generate_calibration_results(dataset, signal, model_type):
    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    generic_data, calibration_data, test = get_data(dataset, signal)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)
    X_test = test[features]
    y_test = test[target]
    calibration_sample_size =utils. get_calibration_sample_sizes(dataset=dataset)
    models =utils. get_model(model_type)
    for clf in models:
        model_name = type(clf).__name__
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "calibration", model_name))

        for size in calibration_sample_size:
            # Get the calibration samples from the unseen subject data
            calibration_samples = calibration_data.sample(n=size, random_state=0)
            clf = utils.get_pipeline_model(clf)
            clf = calibrate_model(model=clf, generic_samples=generic_data,
                                  calibration_samples=calibration_samples, target=target,
                                  features=features)

            # Validate the performance on the validation_samples dataset
            predictions = clf.predict(X_test)
            result= utils.get_prediction_metrics(model_type, predictions=predictions, y_test=y_test)
            print(result.transpose())
            # The output file name is the average samples per subjects (there are 4 subjects)
            result.to_csv(os.path.join(out_dir, str(int(size / 4)) + ".csv"))




