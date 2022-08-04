import os
import pandas as pd
import paths, utils
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def get_calibration_results(dataset, signal, model_type):
    base_model = get_base_model(dataset, signal, model_type)

    model_name = type(base_model).__name__

    subject_id_col = get_subject_id_column(dataset)
    df_generic,df_callibration,df_test = get_data(dataset,signal)
    subject_ids = list(set(df_test[subject_id_col]))

    new_model = build_calibrated_model(base_model.model,model_type)

    target = get_prediction_target(dataset=dataset, model_type=model_type)
    features = get_important_features(dataset=dataset, signal=signal, model_type=model_type)

    for s in subject_ids:
        out_dir = os.path.join(RESULT_PATH, "model-performance",model_type, dataset, signal, "calibration",
                                                                model_name,f'subject_{int(s)}')

        cal = df_callibration[df_callibration[subject_id_col]==s]   
        test = df_test[df_test[subject_id_col]==s]

        X_cal = scale_data(cal[features])
        y_cal = label_encode_target(cal[target])

        X_test = scale_data(test[features])
        y_test = test[target]
        for sample_size in range(10,101,10):
            new_model.fit(X_cal[:sample_size],y_cal[:sample_size],steps_per_epoch= 10, epochs= 500,verbose=0)
            predictions = [list(val).index(max(val)) for val in new_model.predict(X_test)]
            result = get_prediction_metrics(model_type,predictions,y_test)
            result.to_csv(os.path.join(out_dir,f"{sample_size}.csv"), index=True)




