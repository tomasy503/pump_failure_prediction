# import libraries
import os

import pandas as pd
from src.data.preprocessing_functions import preprocess_data

if __name__ == "__main__":
    # workspace_dir = os.environ.get(
    #     "WORKSPACE_DIR", "/workspaces/pump_failure_detection/")
    # data_path = os.path.join(workspace_dir, "data/")

    # # load data
    # df = pd.read_csv(data_path + "raw/pump_sensor.csv", index_col=0)

    # # preprocess data
    # df = preprocess_data(df)

    # # save data
    # df.to_csv(data_path + "processed/preprocessed_data.csv", index=False)
