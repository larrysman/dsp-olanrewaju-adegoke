from typing import Tuple, List
import joblib
import numpy as np
import pandas as pd
dataPATH = '/Users/OLALYTICS/dsp-olanrewaju-adegoke/data/train.csv'


def make_prediction(test_set: pd.DataFrame) -> np.ndarray:
    LinReg_model = joblib.load('models/LinReg.joblib')
    predictions = LinReg_model.predict(test_set)
    return predictions[:5]


def dataframe_equality_check_test_data(test_set: pd.DataFrame) -> pd.DataFrame:
    final_test_csv, final_test_csv_df = save_and_load_parquet_test_csv(
        test_set)
    final_test_csv, final_test_csv_df = resetting_index_of_dataframe(
        final_test_csv,  final_test_csv_df)
    pd.testing.assert_frame_equal(final_test_csv_df, final_test_csv)


def selecting_features_columns_test_data() -> Tuple[List[str], List[str]]:
    categorical_features = ['MSZoning', 'HouseStyle']
    continuous_features = ['YearBuilt', 'TotalBsmtSF', 'MiscVal']
    return categorical_features, continuous_features


def load_test_data() -> pd.DataFrame:
    test_csv_master = pd.read_csv(dataPATH)
    test_csv = test_csv_master.copy()
    return test_csv


def resetting_index_of_dataframe(
        final_test_csv: pd.DataFrame,
        final_test_csv_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    final_test_csv_df = final_test_csv_df.reset_index(drop=True)
    final_test_csv = final_test_csv.reset_index(drop=True)
    return final_test_csv, final_test_csv_df


def save_and_load_parquet_test_csv(
        test_set: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    final_test_csv = test_set
    final_test_csv.to_parquet(
        dataPATH + 'final_test_csv_df.parquet', index=False)
    final_test_csv_df = pd.read_parquet(dataPATH + 'final_test_csv_df.parquet')
    return final_test_csv, final_test_csv_df


def preprocessing_test_data(
        categorical_features: List[str],
        continuous_features: List[str],
        test_csv_features: pd.DataFrame
) -> pd.DataFrame:
    check_and_correct_NaN(test_csv_features)
    test_csv_cat_DF = encoding_categorical_features_test_data(
        categorical_features, test_csv_features)
    test_csv_cont_DF = scaling_continuous_features_test_data(
        continuous_features, test_csv_features)
    test_set = pd.concat([test_csv_cont_DF, test_csv_cat_DF], axis=1)
    return test_set


def scaling_continuous_features_test_data(
        continuous_features: List[str], test_csv_features: pd.DataFrame
) -> pd.DataFrame:
    loaded_stdScaler = joblib.load(
        'models/stdScaler.joblib')
    test_csv_cont = loaded_stdScaler.transform(
        test_csv_features[continuous_features])
    test_csv_cont_DF = pd.DataFrame(
        test_csv_cont, columns=continuous_features)
    return test_csv_cont_DF


def encoding_categorical_features_test_data(
        categorical_features: List[str], test_csv_features: pd.DataFrame
) -> pd.DataFrame:
    loaded_oneHot = joblib.load(
        'models/oneHot.joblib')
    test_csv_cat = loaded_oneHot.transform(
        test_csv_features[categorical_features])
    test_csv_cat_DF = pd.DataFrame(
        test_csv_cat,
        columns=loaded_oneHot.get_feature_names(categorical_features)
        )
    return test_csv_cat_DF


def check_and_correct_NaN(
        test_csv_features: pd.DataFrame
) -> None:
    test_csv_features.isna().sum()
    test_csv_features.dropna(inplace=True)


def make_predictions(dataPath: str) -> np.ndarray:
    test_csv = load_test_data()
    categorical_features, continuous_features = (
        selecting_features_columns_test_data())
    test_csv_features = test_csv[categorical_features + continuous_features]
    test_set = preprocessing_test_data(
        categorical_features, continuous_features, test_csv_features)
    dataframe_equality_check_test_data(test_set)
    predictions = make_prediction(test_set)
    return predictions
