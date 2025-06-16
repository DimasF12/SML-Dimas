import pandas as pd
import argparse
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Parsing parameter untuk file data
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="heart_processed.csv")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.data_path)
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning manual
n_estimators_list = [50, 100, 150]
max_depth_list = [3, 5, 7]

for n in n_estimators_list:
    for depth in max_depth_list:
        with mlflow.start_run():
            # Logging parameter
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)

            # Model training
            model = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)

            # Predict & evaluation
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            # Logging metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            # Logging model
            mlflow.sklearn.log_model(model, "model")

            print(f"Model (n_estimators={n}, max_depth={depth}) → acc={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}")
