# import libraries
import argparse
import os

import pandas as pd
from azureml.core import Run
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.training_functions import (evaluate_preds, model_optimization,
                                          plot_predictions, split_data,
                                          train_model)

if __name__ == "__main__":
    # Get parameters
    run = Run.get_context()

    parser = argparse.ArgumentParser()
    # parser.add_argument("--load_folder", type=str, dest="load_folder")
    parser.add_argument("--train_folder", type=str, dest="train_folder")
    parser.add_argument("--prediction_folder", type=str,
                        dest="prediction_folder")

    args = parser.parse_args()

    os.makedirs(args.prediction_folder, exist_ok=True)

    df = pd.read_csv("{}/df_for_training.csv".format(args.train_folder))

    train_data, test_data, train_target, test_target = split_data(df)

    # Define the class weights
    class_weights = train_target.value_counts().to_dict()
    for key in class_weights.keys():
        class_weights[key] = len(train_target) / class_weights[key]

    # # Define the models to test
    # models = [LGBMClassifier(), RandomForestClassifier(), LogisticRegression()]

    # # Train and evaluate the models
    # plot_predictions(models, train_data, train_target, test_data, test_target)

    # Define the model
    model = LGBMClassifier(class_weight=class_weights)

    predictions = model_optimization(
        model, train_data, train_target, test_data, test_target)

    # # Define parameters after optimization
    # best_params = {
    #     'bagging_fraction': 0.7530345877103315,
    #     'feature_fraction': 0.5,
    #     'learning_rate': 0.23579587919573766,
    #     'max_depth': 10,
    #     'min_child_weight': 1,
    #     'min_split_gain': 0.0,
    #     'n_estimators': 123,
    #     'num_leaves': 92,
    #     'reg_alpha': 10.0,
    #     'reg_lambda': 10.0,
    # }
    # model.set_params(**best_params)

    # predictions = train_model(model, train_data, train_target, test_data)

    # metrics = evaluate_preds(test_target, predictions)

    test_data["predicted_status"] = predictions

    test_data.to_csv(os.path.join(args.prediction_folder,
                     "predictions.csv"), index=False)
