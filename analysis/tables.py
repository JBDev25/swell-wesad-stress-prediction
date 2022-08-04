import pandas as pd

import numpy as np
import paths, utils
import os


def generate_person_specific_result_table(dataset, signal, model_type):
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)
    if model_type == "classification":
        generate_classification_person_spec_table(dataset, model_type, signal)
    else:
        generate_regression_person_spec_table(dataset, model_type, signal)


def generate_regression_person_spec_table(dataset, model_type, signal):
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)
    subject_ids = utils.get_subjects_ids(dataset=dataset)
    subjects = ["subject_" + str(x) for x in subject_ids]
    n_folds = 10
    out_df = pd.DataFrame(columns=subjects, index=np.arange(n_folds))
    models = utils.get_model(model_type="regression")
    temp_df = pd.read_csv(os.path.join(os.path.join(paths.result_directory(), "model-performance", model_type,
                                                    dataset, signal, "person-specific", "ExtraTreesRegressor",
                                                    "subject_2.csv")))
    columns = [x for x in list(temp_df) if x not in ["fit_time", "score_time"]]

    for clf in models:
        model_name = type(clf).__name__
        in_dir = os.path.join(paths.result_directory(), "model-performance", model_type,
                              dataset, signal, "person-specific", model_name)
        file_name = dataset + "-" + signal + "-" + model_type + ".xlsx"

        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables", model_type,
                                                             dataset, signal, "person-specific", model_name))

        writer = pd.ExcelWriter(os.path.join(out_dir, file_name))

        for col in columns:
            sheet_name = col
            for subject in subjects:
                in_file = os.path.join(in_dir, subject + ".csv")
                values = pd.read_csv(in_file)[col]
                if col == 'test_neg_mean_squared_error':
                    values = np.sqrt(-values)
                    sheet_name = "rmse"
                if col == 'test_neg_mean_absolute_error':
                    values = -values
                    sheet_name = "mae"
                out_df[subject] = values

            out_df.index.name = 'cv fold'
            mean = out_df.mean()
            std = out_df.std()
            out_df.loc['mean'] = mean
            out_df.loc['std'] = std
            # Drop empty columns
            out_df.dropna(how='all', axis=1, inplace=True)
            out_df.to_excel(writer, sheet_name=sheet_name, index=True)
        writer.save()


def generate_classification_person_spec_table(dataset, model_type, signal):
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)

    subject_ids = utils.get_subjects_ids(dataset=dataset)
    subjects = ["subject_" + str(x) for x in subject_ids]
    n_folds = 10
    out_df = pd.DataFrame(columns=subjects, index=np.arange(n_folds))
    models = utils.get_model(model_type="classification")
    temp_df = pd.read_csv(os.path.join(paths.result_directory(), "model-performance", model_type,
                                       dataset, signal, "person-specific", "ExtraTreesClassifier", "subject_2.csv"))
    columns = [x for x in list(temp_df) if x not in ["fit_time", "score_time"]]
    for clf in models:
        model_name = type(clf).__name__
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables", model_type,
                                                             dataset, signal, "person-specific", model_name))
        file_name = dataset + "-" + signal + "-" + model_type + ".xlsx"
        writer = pd.ExcelWriter(os.path.join(out_dir, file_name))
        for col in columns:
            for subject in subjects:
                in_dir = os.path.join(paths.result_directory(), "model-performance", model_type,
                                      dataset, signal, "person-specific", model_name)
                in_file = os.path.join(in_dir, subject + ".csv")
                values = pd.read_csv(in_file)[col] * 100
                out_df[subject] = values

            out_df.index.rename('Fold', inplace=True)
            mean = out_df.mean()
            std = out_df.std()
            out_df.loc['mean'] = mean
            out_df.loc['std'] = std
            out_df.to_excel(writer, sheet_name=col, index=True)
        writer.save()


def generate_generic_result_table(dataset, signal, model_type):
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)
    if model_type == "classification":
        generate_generic_classification_table(dataset, model_type, signal)
    else:
        generate_generic_regression_table(dataset, model_type, signal)


def generate_generic_regression_table(dataset, model_type, signal):
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)
    columns = ["MAE", "RMSE"]
    in_dir = os.path.join(paths.result_directory(), "model-performance", model_type,
                          dataset, signal, "generic-model")
    subject_ids = utils.get_subjects_ids(dataset=dataset)
    subjects = ["subject_" + str(x) for x in subject_ids]
    out_df = pd.DataFrame(columns=columns, index=subjects)
    models = utils.get_model(model_type="regression")
    for clf in models:
        model_name = type(clf).__name__
        for subject in subjects:
            in_file = os.path.join(in_dir, model_name, subject + ".csv")
            data = pd.read_csv(in_file, index_col=0)
            for i in columns:
                out_df.at[subject, i] = data.loc[i].values[0]
        out_df.dropna(inplace=True, axis="index")
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables",
                                                             model_type, dataset, signal, "generic", model_name))
        file_name = dataset + "-" + signal + "-" + model_type + ".csv"
        out_df.to_csv(os.path.join(out_dir, file_name), index=True)


def generate_generic_classification_table(dataset, model_type, signal):
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)
    root_dir = os.path.join(paths.result_directory(), "model-performance",
                            model_type, dataset, signal, "generic-model")
    temp_df = pd.read_csv(os.path.join(root_dir, "ExtraTreesClassifier", "subject_2.csv"))
    index = temp_df[list(temp_df)[0]].tolist()
    subject_ids = utils.get_subjects_ids(dataset=dataset)
    subjects = ["subject_" + str(x) for x in subject_ids]
    out_df = pd.DataFrame(columns=subjects, index=index)
    models = utils.get_model(model_type="classification")
    for clf in models:
        model_name = type(clf).__name__
        for subject in subjects:
            in_file = os.path.join(root_dir, model_name, subject + ".csv")
            data = pd.read_csv(in_file)
            out_df[subject] = data[list(data)[1]].values
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables",
                                                             "classification", dataset, signal, "generic", model_name))
        # Drop empty columns
        out_df.dropna(how='all', axis=1, inplace=True)
        out_df.index.name = 'valuation metrics'
        file_name = dataset + "-" + signal + "-" + model_type + ".csv"
        out_df.to_csv(os.path.join(out_dir, file_name), index=True)


def generate_calibration_table(dataset, model_type, signal):
    utils.validate_model_type_name(model_type=model_type)
    utils.validate_dataset_name(dataset=dataset)
    utils.validate_signal_name(signal=signal)
    if model_type == "classification":
        generate_calibration_classification_table(dataset, model_type, signal)
    else:
        generate_calibration_regression_table(dataset, model_type, signal)


def generate_calibration_regression_table(dataset, model_type, signal):
    calibration_samples = utils.get_calibration_sample_sizes(dataset=dataset) / 4
    calibration_samples = [int(x) for x in calibration_samples]
    in_dir = os.path.join(paths.result_directory(), "model-performance", model_type,
                          dataset, signal, "calibration")
    index = ["MAE", "RMSE"]
    out_df = pd.DataFrame(columns=calibration_samples, index=index)
    models = utils.get_model(model_type="regression")
    for clf in models:
        model_name = type(clf).__name__
        if "ExtraTrees" not in model_name:
            continue
        for sample in calibration_samples:
            in_file = os.path.join(in_dir, model_name, str(sample) + ".csv")
            data = pd.read_csv(in_file, index_col=0)
            out_df[sample] = data[list(data)[0]]
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables",
                                                             model_type, dataset, signal, "calibration", model_name))
        file_name = dataset + "-" + signal + "-" + model_type + ".csv"
        out_df.to_csv(os.path.join(out_dir, file_name), index=True)


def generate_calibration_classification_table(dataset, model_type, signal):
    root_dir = os.path.join(paths.result_directory(), "model-performance", model_type,
                            dataset, signal, "calibration")
    temp_df = pd.read_csv(os.path.join(root_dir, "ExtraTreesClassifier", "0.csv"))
    index = temp_df[list(temp_df)[0]].tolist()
    calibration_samples = utils.get_calibration_sample_sizes(dataset=dataset) / 4
    calibration_samples = [int(x) for x in calibration_samples]
    out_df = pd.DataFrame(columns=calibration_samples, index=index)
    models = utils.get_model(model_type="classification")
    for clf in models:
        model_name = type(clf).__name__
        if "ExtraTrees" not in model_name:
            continue
        for sample in calibration_samples:
            in_file = os.path.join(root_dir, model_name, str(sample) + ".csv")
            data = pd.read_csv(in_file, index_col=0)
            out_df[sample] = data[list(data)[0]]
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables",
                                                             model_type, dataset, signal, "calibration", model_name))
        file_name = dataset + "-" + signal + "-" + model_type + ".csv"
        out_df.to_csv(os.path.join(out_dir, file_name), index=True)


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    datasets = ["swell", "wesad"]
    signals = ["hrv", "eda"]
    model_types = ["classification", "regression"]
    #model_types = ["regression"]

    for dataset in datasets:
        for signal in signals:
            for model_type in model_types:
                generate_calibration_regression_table(dataset=dataset, signal=signal, model_type=model_type)

