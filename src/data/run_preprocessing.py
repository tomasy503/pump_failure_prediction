# import libraries
import argparse
import os

import pandas as pd
from azureml.core import Run
from preprocessing_functions import preprocess_data

if __name__ == "__main__":
    # Get parameters
    run = Run.get_context()

    parser = argparse.ArgumentParser()
    # parser.add_argument("--load_folder", type=str, dest="load_folder")
    parser.add_argument("--train_folder", type=str, dest="train_folder")

    args = parser.parse_args()

    os.makedirs(args.train_folder, exist_ok=True)

    print(run.input_datasets["pump_sensor"])

    # Load the dataset
    df = run.input_datasets["pump_sensor"].to_pandas_dataframe(
    )

    # preprocess data
    df = preprocess_data(df)

    # save data
    df.to_csv(
        os.path.join(args.train_folder, "df_for_training.csv"), index=False
    )
