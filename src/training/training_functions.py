import matplotlib.pyplot as plt
import pandas as pd
# Metrics
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


def split_data(data: pd.DataFrame):
    # convert the "timestamp" column to datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Sort the dataframe by 'timestamp'
    data = data.sort_values(by="timestamp")

    # Define mask for training and test sets in real world scenario for production
    mask_train = data["timestamp"] < "2018-07-08 00:53:00"
    mask_test = data["timestamp"] >= "2018-07-08 00:53:00"

    # Create training and test sets
    train_data = data[mask_train]
    test_data = data[mask_test]

    train_data.drop("timestamp", axis=1, inplace=True)
    test_data.drop("timestamp", axis=1, inplace=True)

    train_data["machine_status"] = train_data["machine_status"].astype(int)

    # define target variables
    train_target = train_data["machine_status"]
    test_target = test_data["machine_status"]

    train_data.drop("machine_status", axis=1, inplace=True)
    test_data.drop("machine_status", axis=1, inplace=True)

    return train_data, test_data, train_target, test_target


def train_model(model, train_data, train_target, test_data):
    model.fit(train_data, train_target)
    predictions = model.predict(test_data)
    return predictions


def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_preds labels on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average="macro")
    recall = recall_score(y_true, y_preds, average="macro")
    f1 = f1_score(y_true, y_preds, average="macro")
    metric_dict = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
    }

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 score: {f1 * 100:.2f}%")

    # Confusion matrix
    confusion_matrix = pd.crosstab(y_true, y_preds, rownames=[
                                   "Actual"], colnames=["Predicted"])
    print("Confusion matrix:")
    print(confusion_matrix)

    # Evaluate per class metrics
    class_metrics = classification_report(y_true, y_preds, output_dict=True)
    print("Per class metrics:")
    for class_label, metrics in class_metrics.items():
        if isinstance(metrics, dict):  # Skip entries that are not dictionaries
            print(f"For class {class_label}:")
            print(f"\tPrecision: {metrics['precision'] * 100:.2f}%")
            print(f"\tRecall: {metrics['recall'] * 100:.2f}%")
            print(f"\tF1 score: {metrics['f1-score'] * 100:.2f}%")


def plot_predictions(models, train_data, train_target, test_data, test_target):
    for model in models:
        model_name = type(model).__name__
        print(f"Training and predicting with {model_name}...")
        predictions = train_model(model, train_data, train_target, test_data)
        evaluate_preds(test_target, predictions)

        plt.figure(figsize=(10, 6))
        plt.plot(test_target.values, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.title(f"{model_name} Predictions vs Actual")
        plt.xlabel("Samples")
        plt.ylabel("Machine Status")
        plt.legend()
        plt.show()


def model_optimization(model, train_data, train_target, test_data, test_target):
    # Define search spaces for Bayesian optimization for LightGBM
    # search_spaces = {
    #     # Maximum tree leaves for base learners
    #     'num_leaves': Integer(5, 1000),
    #     # Maximum tree depth for base learners, <=0 means no limit
    #     'max_depth': Integer(1, 15),
    #     'n_estimators': Integer(10, 400),
    #     'learning_rate': Real(0.01, 0.5),
    #     'feature_fraction': Real(0.1, 0.9),
    #     'bagging_fraction': Real(0.8, 1.0),
    #     'min_split_gain': Real(0.01, 0.1),
    #     'min_child_weight': Integer(3, 50),
    #     # L2 regularization
    #     'reg_lambda': Real(0.0001, 10.0, 'log-uniform'),
    #     # L1 regularization
    #     'reg_alpha': Real(0.0001, 10.0, 'log-uniform'),
    # }

    # Define search spaces for Bayesian optimization for LightGBM
    # search_spaces = {
    #     'num_leaves': Integer(5, 256),
    #     'max_depth': Integer(3, 12),
    #     'n_estimators': Integer(50, 300),
    #     'learning_rate': Real(0.01, 0.3),
    #     'feature_fraction': Real(0.5, 0.9),
    #     'bagging_fraction': Real(0.6, 0.9),
    #     'min_split_gain': Real(0.0, 0.1),
    #     'min_child_weight': Integer(1, 30),
    #     'reg_lambda': Real(1e-3, 10.0, 'log-uniform'),
    #     'reg_alpha': Real(1e-3, 10.0, 'log-uniform'),
    # }

    # Define search spaces for Bayesian optimization for XGBoost
    search_spaces = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 12),
        'learning_rate': Real(0.01, 0.3),
        'subsample': Real(0.5, 0.9),
        'colsample_bytree': Real(0.5, 0.9),
        'colsample_bylevel': Real(0.5, 0.9),
        'min_child_weight': Integer(1, 30),
        'reg_lambda': Real(1e-3, 10.0, 'log-uniform'),
        'reg_alpha': Real(1e-3, 10.0, 'log-uniform'),
    }

    # bayes_search = BayesSearchCV(
    #     model, search_spaces, n_iter=50, cv=5)

    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        scoring='roc_auc',
        cv=5,
        n_iter=50
    )

    bayes_search.fit(train_data, train_target)

    best_params = bayes_search.best_params_

    # Update the model with the best parameters
    model.set_params(**best_params)

    # Train the model
    model.fit(train_data, train_target)

    # Predict the test data
    predictions = model.predict(test_data)

    # Evaluate the predictions
    evaluate_preds(test_target, predictions)

    # Print the best parameters after optimization
    print("Best parameters:", best_params)
    return predictions
