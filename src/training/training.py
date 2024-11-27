# import libraries
import os

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.training.training_functions import (evaluate_preds,
                                             model_optimization,
                                             plot_predictions, split_data,
                                             train_model)

if __name__ == "__main__":
    workspace_dir = os.environ.get(
        "WORKSPACE_DIR", "/workspaces/pump_failure_detection/")
    data_path = os.path.join(workspace_dir, "data/")

    # Read processed data
    df = pd.read_csv(data_path + "processed/preprocessed_data.csv")

    train_data, test_data, train_target, test_target = split_data(df)

    # Define the class weights
    class_weights = train_target.value_counts().to_dict()
    for key in class_weights.keys():
        class_weights[key] = len(train_target) / class_weights[key]

    # ############## 1.0 TEST MULTIPLE MODELS ################
    # # Define the models to test
    # models = [LGBMClassifier(), XGBClassifier(), LogisticRegression(), RandomForestClassifier()]

    # # Train and evaluate the models
    # plot_predictions(models, train_data, train_target, test_data, test_target)
    # ########################################################

    # ############## 2.0 OPTIMIZE THE BEST MODEL ################
    # # Define the model
    # model = XGBClassifier()

    # predictions = model_optimization(
    #     model, train_data, train_target, test_data, test_target)
    # ###########################################################

    ############### 3.0 MODEL AFTER OPTIMIZATION ################
    # Define the model
    # model = LGBMClassifier(bagging_fraction=0.7530345877103315, class_weight=class_weights, feature_fraction=0.5,
    #                        learning_rate=0.23579587919573766, max_depth=10, min_child_weight=1, n_estimators=123, num_leaves=92, reg_alpha=10.0, reg_lambda=10.0)

    # Define the model
    # model = XGBClassifier(colsample_bylevel=0.6649909533581455, colsample_bytree=0.5323172223176906, learning_rate=0.23222329475388573,
    #                       max_depth=3, min_child_weight=23, n_estimators=50, reg_alpha=10.0, reg_lambda=1.4144394477389288, subsample=0.5)

    # predictions = train_model(model, train_data, train_target, test_data)

    # metrics = evaluate_preds(test_target, predictions)

    # test_data["actual_status"] = test_target
    # test_data["predicted_status"] = predictions

    # test_data.to_csv(data_path + "interim/predictions.csv", index=False)
    ##############################################################

    #################### 4.0 USE VOTING ENSEMBLE WITH BOTH MODELS ####################
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
