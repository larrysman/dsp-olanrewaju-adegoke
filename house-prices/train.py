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

    # Loading the train.csv dataset from path
    dataset = load_selected_dataset(dataPATH)

    # Selecting the categorical and continuous columns of interest
    categorical_features, continuous_features = select_features_columns()

    # Defining the features and target
    features, target = features_target_selection(dataset, categorical_features, continuous_features)

    # Splitting of the dataset into train and test sets
    X_train, X_test, y_train, y_test = custom_splitting(features, target)

    # Preprocessing and feature engineering of the X_train set
    X_train, X_test = preprocessing(categorical_features, continuous_features, X_train, X_test)

    # Automatic checking of the DataFrame Equality for X_train dataset
    X_train, X_test = DataFrame_Equality(dataPATH, X_train, X_test)

    # Model training and fitting
    LinReg = training_model(X_train, y_train)

    # Model predictions of the X_test
    y_predicted = model_prediction(X_test)

    # Model evaluation and model performance
    rmsle = model_performance(y_test, y_predicted)

    return LinReg, rmsle


