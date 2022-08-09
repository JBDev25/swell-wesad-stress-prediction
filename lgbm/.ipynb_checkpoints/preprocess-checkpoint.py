import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer


def preprocess_data(data_frame,target_var,features, subject=None):
    x,y = get_subject_features_and_target(data_frame,subject=subject,target_var=target_var,features=features)
    X=StandardScaler().fit_transform(x)
    return X, y

    
def get_subject_features_and_target(df,target_var,features, subject=None):
    sub_df = df.copy()
    if subject is not None:
        sub_df = sub_df[sub_df['subject_id']==subject]
    x_cal = sub_df[features].values
    y_cal = sub_df[target_var].values
    return x_cal,y_cal 
