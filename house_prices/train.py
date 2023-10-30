import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error


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


def preprocessing(categorical_features: List[str],
        continuous_features: List[str], X_train: pd.DataFrame,
        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = preprocessing_x_train(categorical_features,
        continuous_features, X_train)
    X_test = preprocessing_x_test(categorical_features,
        continuous_features, X_test)
    return X_train,X_test


def training_model(
        X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    LinReg = LinearRegression()
    LinReg.fit(X_train, y_train)
    return LinReg
    joblib.dump(LinReg, '../models/LinReg.joblib')


def model_prediction(X_test: pd.Series) -> np.ndarray:
    y_predicted = LinReg_model.predict(X_test)
    return y_predicted
    LinReg_model = joblib.load('../models/LinReg.joblib')


def model_performance(y_test: np.ndarray, 
        y_predicted: np.ndarray) -> dict[str, Union[str, float]]:
    msle = mean_squared_log_error(y_test, y_predicted)
    rmsle = round(np.sqrt(msle), 2)
    model_rmsle = {'rmsle' : rmsle}
    return model_rmsle


def build_model(dataPATH: str) -> Tuple[
        LinearRegression, Dict[str, float]]:
    dataset = load_selected_dataset(dataPATH)
    categorical_features, continuous_features = select_features_columns()
    features, target = features_target_selection(
        dataset, categorical_features, continuous_features)
    X_train, X_test, y_train, y_test = custom_splitting(features, target)
    X_train, X_test = preprocessing(categorical_features, 
        continuous_features, X_train, X_test)
    X_train, X_test = DataFrame_Equality(dataPATH, X_train, X_test)
    LinReg = training_model(X_train, y_train)
    y_predicted = model_prediction(X_test)
    rmsle = model_performance(y_test, y_predicted)
    return LinReg, rmsle
