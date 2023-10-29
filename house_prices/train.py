import numpy as np
import pandas as pd
import jobliib

def training_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    LinReg = LinearRegression()
    LinReg.fit(X_train, y_train)
    joblib.dump(LinReg, '../models/LinReg.joblib')
    return LinReg

def model_prediction(X_test: pd.Series) -> np.ndarray:
    LinReg_model = joblib.load('../models/LinReg.joblib')
    y_predicted = LinReg_model.predict(X_test)
    return y_predicted

def model_performance(y_test: np.ndarray, y_predicted: np.ndarray) -> dict[str, Union[str, float]]:
    msle = mean_squared_log_error(y_test, y_predicted)
    rmsle = round(np.sqrt(msle), 2)
    model_rmsle = {'rmsle' : rmsle}
    return model_rmsle

def build_model(dataPATH: str) -> Tuple[LinearRegression, Dict[str, float]]:
    dataset = load_selected_dataset(dataPATH)
    categorical_features, continuous_features = select_features_columns()
    features, target = features_target_selection(dataset, categorical_features, continuous_features)
    X_train, X_test, y_train, y_test = custom_splitting(features, target)
    X_train, X_test = preprocessing(categorical_features, continuous_features, X_train, X_test)
    X_train, X_test = DataFrame_Equality(dataPATH, X_train, X_test)
    LinReg = training_model(X_train, y_train)
    y_predicted = model_prediction(X_test)
    rmsle = model_performance(y_test, y_predicted)

    return LinReg, rmsle
