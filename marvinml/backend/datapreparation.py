from marvinml.backend.utils import one_hot_encoder, label_encoder
from marvinml.backend.utils import minmax, standard_scaler, normalizer
from marvinml.backend.utils import imputation_mean, imputation_median, imputation_deletion_case
from marvinml.backend.utils import oversampling, undersampling
from marvinml.backend.utils import separate
import pandas as pd
import numpy as np

PIPELINE_OPTIONS = {
    'imputation_mean': imputation_mean,
    'imputation_median': imputation_median,
    'imputation_deletion_case': imputation_deletion_case,
    'one_hot_encoder': one_hot_encoder,
    'label_encoder': label_encoder,
    'minmax': minmax,
    'normalizer': normalizer,
    'standard_scaler': standard_scaler,
    'oversampling': oversampling,
    'undersampling': undersampling,
}

def transform(dataframe, trans):
    num, cat, y_target, num_columns, cat_columns = separate(dataframe)
    if type(cat) is pd.Series:
        cat = pd.DataFrame(cat, columns = cat_columns)
    if cat is not None:
        data = trans.transform(cat)
        data = pd.DataFrame(data, columns = cat_columns)
        if num is not None:
            data = pd.concat([num, data], axis=1)
    else: 
        data = num
    data = pd.concat([data, y_target], axis=1)
    return data

def preprocess(dataframe, pipeline, target, **kwargs):
    transformer = None
    for stage in pipeline:
        if stage in PIPELINE_OPTIONS:
            if stage is 'label_encoder':
                print("Stage --> ", stage)
                dataframe, transformer = PIPELINE_OPTIONS[stage](dataframe, target)
            else:
                print("Stage --> ", stage)
                dataframe = PIPELINE_OPTIONS[stage](dataframe, target)
    return dataframe, transformer

def data_preparation(dataframe, target, description=False, **kwargs):
    if(description):
        print("Case 1: ['no_preparation']")
        print('Dataset without categorical data and null values.')
        print("In this case, no imputation or transformation is required. \n") 
        
        print("Case 2: ['imputation_median']")
        print("Dataset without categorical data but have null values in your set.")
        print("In this case, only the imputation of null values is necessary. \n") 
        
        print("Case 3: ['label_encoder']")
        print("Dataset without null values but have categorical data in your set.") 
        print("In this case, only the transformation of categorical values is necessary. \n")
        
        print("Case 4: ['imputation_deletion_case', 'label_encoder', 'oversampling']")
        print("Dataset with categorical data and null values.")
        print("This case has null values and categorical data, it takes both imputation and transformation.") 
        print("There are nulls in the categorical data, so the imputation needs to be compatible. \n")
        
        print("Case 5: ['imputation_median', 'label_encoder']")
        print("Dataset with categorical data but have null values only in numeric columns.")
        print("This case has null values and categorical data, it takes both imputation and transformation.") 
        print("There are nulls only in the numerical data, it can be any type of imputation. \n")
        
        print("===================================== \n")
        print("Initializing the algorithm... \n")
   
    total_null = dataframe.isna().sum().sum()
    num_columns, cat_columns, _, _, _ = separate(dataframe, target)
    if(cat_columns is None):
        cat_columns = 0
    else:
        cat_null = cat_columns.isna().sum().sum()
        cat_columns = 1

    if(num_columns is None):
        num_columns = 0
    else:
        num_null = num_columns.isna().sum().sum()
        num_columns = 1

    if((cat_columns == 0) and (total_null == 0)): 
        print('Dataset without categorical data and null values.')
        pipeline = ['no_preparation']
        print('No need data preparation.')

    elif((cat_columns == 0) and (num_null > 0)): 
        print('Dataset without categorical data but have null values in your set.')
        pipeline = ['imputation_median']
        print('Applied techniques: ', pipeline, '\n')

    elif(((cat_columns == 1) and (total_null == 0)) or ((num_columns == 0) and (total_null == 0))):  
        print('Dataset without null values but have categorical data in your set.')
        pipeline = ['label_encoder']
        print('Applied techniques: ', pipeline, '\n')

    elif(((cat_columns == 1) and (cat_null > 0)) or ((num_columns == 0) and (cat_null > 0))):
        print('Dataset with categorical data and null values.')
        pipeline = ['imputation_deletion_case', 'label_encoder', 'oversampling']
        print('Applied techniques: ', pipeline, '\n')

    else:
        print('Dataset with categorical data but have null values only in numeric columns.')
        pipeline = ['imputation_median', 'label_encoder']
        print('Applied techniques: ', pipeline, '\n')
        
    transformer = None
    for stage in pipeline:
        if stage in PIPELINE_OPTIONS:
            if stage is 'label_encoder':
                print("Stage --> ", stage)
                dataframe, transformer = PIPELINE_OPTIONS[stage](dataframe, target)
            else:
                print("Stage --> ", stage)
                dataframe = PIPELINE_OPTIONS[stage](dataframe, target)
    return dataframe, transformer