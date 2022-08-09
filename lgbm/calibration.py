from lightgbm import LGBMClassifier, LGBMRegressor
import pandas as pd
import os

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score 
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
from preprocess import preprocess_data
from utils import get_classification_metrics, get_regression_metrics, get_calibration_subjects


def get_calibration_results(df_generic,df_callibration,df_test,target_var,output_dir,model_type,calibration_subjects):
    
    feat = df_generic.columns.values[:-3]
    
    X_gen, y_gen = preprocess_data(df_generic,target_var,feat)
    X_gen,y_gen = X_gen[:500],y_gen[:500]
    
    for subject in calibration_subjects:
        print(f'======SUBJECT: {subject}======')
        result_f = pd.DataFrame()

        # Preprocess Data    
        X_cal, y_cal = preprocess_data(df_callibration,target_var,feat,subject)
        X_test, y_test = preprocess_data(df_test,target_var,feat,subject)

        # Train Calibration
        for n in range(0,101,10):
            print('Samples:',n)
            # Add Calibration Samples
            X_cal_n = np.append(X_gen,X_cal[:n],axis=0)
            y_cal_n = np.append(y_gen,y_cal[:n],axis=0)

            # Train Model and get results
            if model_type == 'class':
              model = LGBMClassifier()
              model.fit(X_cal_n,y_cal_n)
              pred = model.predict(X_test)
              result = get_classification_metrics(pred,y_test)
              
            else:
              model = LGBMRegressor()
              model.fit(X_cal_n,y_cal_n)
              pred = model.predict(X_test)
              result = get_regression_metrics(pred,y_test)
              
            # Concat Results
            result_f[n] = result
            print("user_id {0} \t{1}".format(subject, result.transpose()))
        # Write Results to csv
        result_f.to_csv(os.path.join(output_dir, "subject_" + str(int(subject)) + ".csv"), index=True)
        
   