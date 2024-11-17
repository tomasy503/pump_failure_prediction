# import libraries
import pandas as pd


# function to preprocess data
def preprocess_data(data):
    # fill missing values
    # convert the "timestamp" column to datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # replace nan values with 0 only on columns starting with sensor_
    data.loc[:, data.columns.str.startswith("sensor_")] = data.loc[
        :, data.columns.str.startswith("sensor_")
    ].fillna(0)

    # Convert "machine_status" column to categorical type
    data["machine_status"] = data["machine_status"].astype("category")

    # Code the categories
    data["machine_status_code"] = data["machine_status"].cat.codes

    # create new time features
    data["timestamp_year"] = data["timestamp"].dt.year
    data["timestamp_month"] = data["timestamp"].dt.month
    data["timestamp_day"] = data["timestamp"].dt.day
    data["timestamp_hour"] = data["timestamp"].dt.hour
    data["timestamp_minute"] = data["timestamp"].dt.minute

    # encode categorical features
    # data = pd.get_dummies(data)
    return data
