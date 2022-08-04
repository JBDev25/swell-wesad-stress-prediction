from models import calibrate
from models import calibrate
from models import generic
from models import person_specific_model
from analysis import tables
from analysis import plots

#datasets = ["swell", "wesad"]
#signals = ["hrv", "eda"]
#model_types = ["classification", "regression"]


datasets = ["swell"]
signals = ["hrv"]
model_types = ["classification"]



def generate_all_calibration_results():
    for dataset in datasets:
        for signal in signals:
            for model_type in model_types:
                calibrate.generate_calibration_results(dataset=dataset, signal=signal, model_type=model_type)


def generate_all_generic_results():
    for dataset in datasets:
        for signal in signals:
            for model_type in model_types:
                generic.generate_generic_results(dataset=dataset, signal=signal, model_type=model_type)


def generate_all_person_specific_results():
    for dataset in datasets:
        for signal in signals:
            for model_type in model_types:
                person_specific_model.generate_person_specific_results(dataset=dataset, signal=signal,
                                                                       model_type=model_type)


# def generate_all_person_spec_result_table():
#     for dataset in datasets:
#         for signal in signals:
#             for model_type in model_types:
#                 tables.generate_person_specific_result_table(dataset=dataset, signal=signal, model_type=model_type)


# def generate_all_generic_result_table():
#     for dataset in datasets:
#         for signal in signals:
#             for model_type in model_types:
#                 tables.generate_generic_result_table(dataset=dataset, signal=signal, model_type=model_type)


# def generate_all_calibration_result_table():
#     for dataset in datasets:
#         for signal in signals:
#             for model_type in model_types:
#                 tables.generate_calibration_table(dataset=dataset, signal=signal, model_type=model_type)


# def generate_comparision_plots():
#     models = ["RandomForest"]
#     for dataset in datasets:
#         for signal in signals:
#             for model_name in models:
#                 plots.plot_generic_vs_person_specific_model(dataset=dataset, signal=signal, model_name=model_name)
#                 #plots.plot_calibration_result(dataset=dataset, signal=signal)

# def generate_calibration_plots():
#     models = ["N"]
#     for dataset in datasets:
#         for signal in signals:
#             for model_name in models:
#                 plots.plot_calibration_result(dataset=dataset, signal=signal, model_name=model_name)

# def generate_combined_general_model_performance():
#     for dataset in datasets:
#         for signal in signals:
#             for model_type in model_types:
#                 print(dataset)
#                 generic.generate_combined_general_results(dataset=dataset, signal=signal, model_type=model_type)




if __name__ == "__main__":
    import warnings

    warnings.simplefilter(action='ignore')
    """
    Note: 
    0. Do not delete the files in the results/feature-ranks folder. I did not include the code to generate it
    1. Everything else in the results folder can be generated by running functions below
    2. Running the first three functions will take many hours to finish. 
    """
    generate_all_person_specific_results()
    generate_all_calibration_results()
    generate_all_generic_results()
    # generate_all_person_spec_result_table()
    # generate_all_generic_result_table()
    # generate_all_calibration_result_table()
    # generate_calibration_plots()
