import numpy as np
import pandas as pd
import jobliib


def load_selected_dataset(dataPATH: str) -> pd.DataFrame:
    train_csv_master = pd.read_csv(dataPATH)
    train_csv = train_csv_master.copy()
    dataset = train_csv[
        ['MSZoning','HouseStyle','YearBuilt','TotalBsmtSF','MiscVal','SalePrice]        ]
    return dataset


def select_features_columns() -> Tuple[List[str], List[str]]:
    categorical_features = ['MSZoning','HouseStyle']
    continuous_features = ['YearBuilt','TotalBsmtSF','MiscVal']
    return categorical_features,continuous_features


def features_target_selection(dataset: pd.DataFrame, 
        categorical_features: List[str], 
        continuous_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    features = dataset[categorical_features + continuous_features]
    target = dataset['SalePrice']
    return features,target


def custom_splitting(features: pd.DataFrame, 
        target: pd.Series, test_size: float=0.3, 
        random_state: int=42) -> Tuple[pd.DataFrame, 
        pd.DataFrame, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, 
        random_state=random_state)
    return X_train,X_test,y_train,y_test


def encoding_categorical_features(categorical_features: List[str], 
        X_train: pd.DataFrame) -> pd.DataFrame:
    oneHot = OneHotEncoder(drop = 'first', sparse=False)
    oneHot.fit(X_train[categorical_features])
    X_train_cat = oneHot.transform(
        X_train[categorical_features])
    X_train_cat_DF = pd.DataFrame(X_train_cat, 
        columns=oneHot.get_feature_names(categorical_features ))
    joblib.dump(oneHot, '../models/oneHot.joblib')
    return X_train_cat_DF


def scaling_continuous_features(continuous_features: List[str], 
        X_train: pd.DataFrame) -> pd.DataFrame:
    stdScaler = StandardScaler()
    stdScaler.fit(X_train[continuous_features])
    X_train_cont = stdScaler.transform(X_train[continuous_features])
    X_train_cont_DF = pd.DataFrame(X_train_cont, columns=continuous_features)
    joblib.dump(stdScaler, '../models/stdScaler.joblib')
    return X_train_cont_DF


def preprocessing_x_train(categorical_features: List[str], 
        continuous_features: List[str], X_train: pd.DataFrame) -> pd.DataFrame:
    X_train_cat_DF = encoding_categorical_features(
        categorical_features, X_train)
    X_train_cont_DF = scaling_continuous_features(
        continuous_features, X_train)
    X_train = pd.concat([X_train_cont_DF, X_train_cat_DF], axis=1)
    return X_train


def save_load_to_parquet_x_train(dataPATH: str, 
        X_train: pd.DataFrame) -> pd.DataFrame:
    X_train.to_parquet(dataPATH + 'X_train_df.parquet', index=False)
    X_train_df = pd.read_parquet(dataPATH + 'X_train_df.parquet')
    return X_train_df


def resetting_index_parquet_x_train(X_train: pd.DataFrame, 
        X_train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_df = X_train_df.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    return X_train,X_train_df


def x_train_dataframe_equality_check(dataPATH: str, 
        X_train: pd.DataFrame) -> pd.DataFrame:
    X_train_df = save_load_to_parquet_x_train(dataPATH, X_train)
    X_train, X_train_df = resetting_index_parquet_x_train(
        X_train, X_train_df)      
    pd.testing.assert_frame_equal(X_train_df, X_train)
    return X_train


def encoding_categorical_features(categorical_features: List[str], 
        X_test: pd.DataFrame) -> pd.DataFrame:
    loaded_oneHot = joblib.load('../models/oneHot.joblib')
    X_test_cat = loaded_oneHot.transform(X_test[categorical_features])
    X_test_cat_DF = pd.DataFrame(X_test_cat, 
        columns = loaded_oneHot.get_feature_names(categorical_features ))
    return X_test_cat_DF


def scaling_continuous_features(continuous_features: List[str], 
        X_test: pd.DataFrame) -> pd.DataFrame:
    loaded_stdScaler = joblib.load('../models/stdScaler.joblib')
    X_test_cont = loaded_stdScaler.transform(X_test[continuous_features])
    X_test_cont_DF = pd.DataFrame(X_test_cont, columns=continuous_features)
    return X_test_cont_DF


def preprocessing_x_test(categorical_features: List[str], 
        continuous_features: List[str], 
        X_test: pd.DataFrame) -> pd.DataFrame:
    X_test_cat_DF = encoding_categorical_features(
        categorical_features, X_test)
    X_test_cont_DF = scaling_continuous_features(
        continuous_features, X_test)
    X_test = pd.concat([X_test_cont_DF, X_test_cat_DF], axis=1)
    return X_test


def save_load_to_parquet_x_test(dataPATH: str, 
        X_test: pd.DataFrame) -> pd.DataFrame:
    X_test.to_parquet(dataPATH + 'X_test_df.parquet', index=False)
    X_test_df = pd.read_parquet(dataPATH + 'X_test_df.parquet')
    return X_test_df


def resetting_index_parquet_x_test(X_test: pd.DataFrame, 
        X_test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_test_df = X_test_df.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_test,X_test_df


def x_test_dataframe_equality_check(dataPATH: str, 
        X_test: pd.DataFrame) -> pd.DataFrame:
    X_test_df = save_load_to_parquet_x_test(dataPATH, X_test)
    X_test, X_test_df = resetting_index_parquet_x_test(X_test, X_test_df)       
    pd.testing.assert_frame_equal(X_test_df, X_test)
    return X_test


def DataFrame_Equality(dataPATH: str, X_train: pd.DataFrame, 
        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = x_train_dataframe_equality_check(dataPATH, X_train)
    X_test = x_test_dataframe_equality_check(dataPATH, X_test)
    return X_train,X_test


def preprocessing(categorical_features: List[str], 
        continuous_features: List[str], X_train: pd.DataFrame, 
        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = preprocessing_x_train(categorical_features, 
        continuous_features, X_train)
    X_test = preprocessing_x_test(categorical_features, 
        continuous_features, X_test)
    return X_train,X_test
