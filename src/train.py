
import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "Product_Store_Sales_Total"

print("Loading training and testing data...")

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_cols),
    (
        "categorical",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_cols
    )
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    )
])

param_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 12, 18],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.8]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=8,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

predictions = best_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

with open(
    "models/metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=2)

print("Best parameters:")
print(search.best_params_)

print("Model Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

print("✅ Model training completed")
