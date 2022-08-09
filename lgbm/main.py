from lightgbm import LGBMClassifier, LGBMRegressor
import pandas as pd
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score 
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
from preprocess import preprocess_data
from utils import get_classification_metrics, extract_dataframe, get_target_var, get_calibration_subjects
from calibration import get_calibration_results

# Extract Data

signals = ['hrv','eda']
datasets = ['swell','wesad']
model_type = ['class','reg']


for signal in signals:
    for dataset in datasets:
        for model in model_type:
            # Set Target and calibration 
            target = get_target_var(dataset,model)
            calibration_subjects = get_calibration_subjects(signal,dataset)
            output_dir = f'../results/{signal}/{dataset}/{model}'
            
            # Extract Data
            df_generic,df_callibration,df_test = extract_dataframe(signal,dataset)
            # Calibration
            # get_calibration_results(df_generic,df_callibration,df_test,
            #                         target,
            #                         output_dir,
            #                         model,
            #                         calibration_subjects)
            # Person Specific
            get_personal_results(df_callibration,df_test,
                                    target,
                                    output_dir,
                                    model,
                                    calibration_subjects)


        
        
        
    
