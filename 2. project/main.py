import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load training and test data (expects CSV files under the 'kaggle/' directory)
train_df = pd.read_csv("kaggle/train.csv", encoding="utf-8")
test_df = pd.read_csv("kaggle/test.csv", encoding="utf-8")

# Keep 'id' columns for later submission and remove them from the feature sets
train_ids = train_df["id"]
test_ids = test_df["id"]
train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)

# Declare categorical features and ensure proper dtypes
categorical_features = ["Sex"]
train_df["Sex"] = train_df["Sex"].astype("category")
test_df["Sex"] = test_df["Sex"].astype("category")

# Separate features and target for training; copy test features for inference
X = train_df.drop(columns=["Age"])
y = train_df["Age"]
X_test = test_df.copy()

# Create a fixed train/validation split for reproducible model selection
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Basic EDA plots for quick inspection (optional)
# Target distribution
plt.figure(figsize=(10, 5))
sns.histplot(y, bins=30, kde=True, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Correlation heatmap among numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(
    train_df.corr(numeric_only=True),
    annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5
)
plt.title("Feature Correlation Matrix")
plt.show()

# Define Optuna objective to minimize validation MAE
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'loss_function': 'MAE',
        'cat_features': categorical_features,
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostRegressor(**params)
    # Early stopping on the validation set to prevent overfitting during search
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=300,
        verbose=0
    )
    y_pred = model.predict(X_valid)
    return mean_absolute_error(y_valid, y_pred)

# Run hyperparameter search
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Train the final model with best parameters (shorter early stopping for faster convergence)
best_params = study.best_params
best_params.update({
    'loss_function': 'MAE',
    'cat_features': categorical_features,
    'random_seed': 42,
    'verbose': 200
})
model = CatBoostRegressor(**best_params)
model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100
)

# Evaluate on the validation set
y_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
print("Validation MAE: {:.4f}".format(mae))

# Visualize predicted vs actual values on the validation set
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_valid, y=y_pred, alpha=0.5, color='red')
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age")
plt.show()

# Predict on the test set and create a submission file
test_predictions = model.predict(X_test)
submission = pd.DataFrame({"id": test_ids, "Age": test_predictions})
submission.to_csv("submission(final).csv", index=False)
print("Submission file saved as submission(final).csv")