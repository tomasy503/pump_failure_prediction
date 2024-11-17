# import libraries
import numpy as np
import pandas as pd
from preprocessing_functions import preprocess_data

if __name__ == "__main__":
    # load data
    df = pd.read_csv("../../data/raw/pump_sensor.csv", index_col=0)

    # preprocess data
    df = preprocess_data(df)

    # save data
    df.to_csv("../data/processed/preprocessed_data.csv", index=False)
