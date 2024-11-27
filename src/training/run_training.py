# import libraries
import argparse
import os

import pandas as pd
from azureml.core import Run
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from training_functions import (evaluate_preds, model_optimization,
                                plot_predictions, split_data, train_model)
from xgboost import XGBClassifier

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

    # ################### OPTIMIZE THE MODEL #####################
    # # Define the model
    # model = XGBClassifier()

    # predictions = model_optimization(
    #     model, train_data, train_target, test_data, test_target)
    # ##########################################################

    ################## MODEL AFTER OPTIMIZATION ###################
    # Define the model
    # model = LGBMClassifier(bagging_fraction=0.7530345877103315, class_weight=class_weights, feature_fraction=0.5,
    #                        learning_rate=0.23579587919573766, max_depth=10, min_child_weight=1, n_estimators=123, num_leaves=92, reg_alpha=10.0, reg_lambda=10.0)

    # # Define the model
    # model = XGBClassifier(colsample_bylevel=0.6649909533581455, colsample_bytree=0.5323172223176906, learning_rate=0.23222329475388573,
    #                       max_depth=3, min_child_weight=23, n_estimators=50, reg_alpha=10.0, reg_lambda=1.4144394477389288, subsample=0.5)

    # predictions = train_model(model, train_data, train_target, test_data)

    # metrics = evaluate_preds(test_target, predictions)

    # test_data["actual_status"] = test_target
    # test_data["predicted_status"] = predictions

    # test_data.to_csv(os.path.join(args.prediction_folder,
    #                  "predictions.csv"), index=False)

    ####################### VOTING ENSEMBLE #########################
    # Define the models
    models = [
        ("LGBM", LGBMClassifier(bagging_fraction=0.7530345877103315, class_weight=class_weights, feature_fraction=0.5,
                                learning_rate=0.23579587919573766, max_depth=10, min_child_weight=1, n_estimators=123, num_leaves=92, reg_alpha=10.0, reg_lambda=10.0)),
        ("XGB", XGBClassifier(colsample_bylevel=0.6649909533581455, colsample_bytree=0.5323172223176906, learning_rate=0.23222329475388573,
                              max_depth=3, min_child_weight=23, n_estimators=50, reg_alpha=10.0, reg_lambda=1.4144394477389288, subsample=0.5))
    ]

    # # Use the voting classifier
    voting_clf = VotingClassifier(
        estimators=models, voting='soft', weights=[1, 2])

    predictions = train_model(voting_clf, train_data, train_target, test_data)

    metrics = evaluate_preds(test_target, predictions)

    test_data["actual_status"] = test_target
    test_data["predicted_status"] = predictions

    test_data.to_csv(os.path.join(args.prediction_folder,
                     "predictions.csv"), index=False)
