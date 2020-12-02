from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from pandas import read_csv, get_dummies

import numpy as np

def separate(data, target=None): 
    all_columns = list(data.columns)
    num_columns = data.select_dtypes(include=np.number).columns.tolist() 
    cat_columns = list(set(all_columns)-set(num_columns)) 
    if target is not None:
        y_target = data[target] 

        if target in num_columns: 
            num_columns.remove(target)
        if target in cat_columns: 
            cat_columns.remove(target)
    else:
        y_target = None
    
    num, cat = None, None
    for column in num_columns: 
        if(num is None):
            num = data[column]
        else:
            num = pd.concat([num, data[column]], axis=1)
    for column in cat_columns: 
        if(cat is None):
            cat = data[column]
        else:
            cat = pd.concat([cat, data[column]], axis=1)
            
    if num is not None:
        num = num.reset_index(drop=True)
    if cat is not None:
        cat = cat.reset_index(drop=True)
    return num, cat, y_target, num_columns, cat_columns

#Apply One-Hot-Encoder
def one_hot_encoder(data, target):

    x = data.drop([target], axis=1)
    y = data[target]
    data = get_dummies(x)
    data = pd.concat([data, y], axis=1)
    return data

#Apply Label Encoder
def label_encoder(data, target=None):

    num, cat, y_target, num_columns, cat_columns = separate(data, target)
    if cat is not None:
        cat = cat.replace(np.nan, 'NaN')
    oe = OrdinalEncoder()
    if type(cat) is pd.Series:
        cat = pd.DataFrame(cat, columns = cat_columns)
    if cat is not None:
        data = oe.fit_transform(cat)
        data = pd.DataFrame(data, columns = cat_columns)
        if num is not None:
            data = pd.concat([num, data], axis=1)
    else: 
        data = num
    data = pd.concat([data, y_target], axis=1)
    return data, oe

####### SCALERS ####### 

#Apply MinMax Scaler
def minmax(data, target):

    num, cat, y_target, num_columns, cat_columns = separate(data, target)
    scaler = MinMaxScaler()
    if num is not None:
        data = scaler.fit_transform(num) 
        data = pd.DataFrame(data, columns=num_columns) 
        if cat is not None:
            data = pd.concat([data, cat], axis=1) 
    else:
        data = cat
    data = pd.concat([data, y_target], axis=1)
    return data

#Apply Standard Scaler
def standard_scaler(data, target):

    num, cat, y_target, num_columns, cat_columns = separate(data, target)
    scaler = StandardScaler()
    if num is not None:
        data = scaler.fit_transform(num) 
        data = pd.DataFrame(data, columns=num_columns) 
        if cat is not None:
            data = pd.concat([data, cat], axis=1) 
    else:
        data = cat
    data = pd.concat([data, y_target], axis=1)
    return data

#Apply Normalizer Scaler
def normalizer(data, target):

    try:
        num, cat, y_target, num_columns, cat_columns = separate(data, target)
        if num is not None:
            num = num.reset_index(drop=True)
            transformer = Normalizer().fit(num)
            data = transformer.transform(num)
            data = pd.DataFrame(data, columns=num_columns)
            if cat is not None:
                cat = cat.reset_index(drop=True)
                data = pd.concat([data, cat], axis=1) 
        data = pd.concat([data, y_target], axis=1)
        return data
    except OSError as err:
        print("OS error: {0}".format(err))
        return data
    except ValueError:
        print("Input contains NaN, infinity or a value too large for dtype('float64').")
        return data
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return data

####### DATA IMPUTATION #######

#Data imputation mean 
def imputation_mean(data, target): 
    
    if data.isna().sum().sum() > 0:
        x = data.drop([target], axis=1)
        y = data[target]
        data = x.fillna(x.mean())
        data = pd.concat([data, y], axis=1)
    return data

#Data imputation median 
def imputation_median(data, target): 

    if data.isna().sum().sum() > 0:
        x = data.drop([target], axis=1)
        y = data[target]
        data = x.fillna(x.median())
        data = pd.concat([data, y], axis=1)
    return data

#Data deletion case 
def imputation_deletion_case(data, target):

    if data.isna().sum().sum() > 0:
        data.dropna(inplace=True)
        data = data.reset_index(drop=True)
    return data

####### IMBALANCED DATASETS #######

def undersampling(data, target):
    under = RandomUnderSampler(random_state=42)
    y = data[target]
    X = data.drop([target], axis=1)
    all_columns = list(X.columns)
    try:
        y = y.to_numpy()
        X = X.to_numpy()
        X, y = under.fit_resample(X, y)
        y = pd.DataFrame(data=y, columns=[target])
        X = pd.DataFrame(data=X, columns=all_columns)
        data = pd.concat([X, y], axis=1)
        return data
    except:
        return data

def oversampling(data, target):
    over = RandomOverSampler(random_state=42)
    y = data[target]
    X = data.drop([target], axis=1)
    all_columns = list(X.columns)
    try:
        X, y = over.fit_resample(X, y)
        y = pd.DataFrame(data=y, columns=[target])
        X = pd.DataFrame(data=X, columns=all_columns)
        data = pd.concat([X, y], axis=1)
        return data
    except:
        return data