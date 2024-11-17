# import libraries
import pandas as pd
from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import (DropConstantFeatures,
                                      DropDuplicateFeatures,
                                      SmartCorrelatedSelection)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# function to preprocess data
def preprocess_data(data):
    # fill missing values
    # convert the "timestamp" column to datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # drop columns with high missing values
    data = data.drop(["sensor_00", "sensor_15", "sensor_50"], axis=1)

    # select sensor columns
    sensor_columns = [col for col in data.columns if col.startswith("sensor_")]

    # fill missing values with the mean of the column
    imputer = MeanMedianImputer(imputation_method="mean")
    data[sensor_columns] = imputer.fit_transform(data[sensor_columns])

    # Convert "machine_status" column to numerical categories
    data["machine_status"] = data["machine_status"].map(
        {"NORMAL": 0, "BROKEN": 1, "RECOVERING": 1, })

    # Create a pipeline to drop constant and duplicate features and select correlated features
    pipeline = Pipeline([
        ("constant_features", DropConstantFeatures(tol=0.95)),
        ("duplicated_features", DropDuplicateFeatures()),
        ("correlated_features", SmartCorrelatedSelection()),
    ])

    # Apply the pipeline to the sensor columns
    data_transformed = pipeline.fit_transform(data[sensor_columns])

    sensor_columns = data_transformed.columns

    # Scale features for sensor data
    scaler = MinMaxScaler()
    data_transformed = scaler.fit_transform(data_transformed)

    data_transformed = pd.DataFrame(
        data_transformed, columns=sensor_columns)

    # Include the timestamp and machine_status columns
    data_final = pd.concat(
        [data[["timestamp", "machine_status"]], data_transformed], axis=1)

    # move the machine_status column to the last column
    data_final = data_final[[
        col for col in data_final.columns if col != "machine_status"] + ["machine_status"]]

    # Return the preprocessed data
    return data_final
