import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("heart_processed.csv")
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tuning manual
for n in [50, 100, 150]:
    for depth in [5, 10, 15]:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Manual logging
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)
            
            mlflow.sklearn.log_model(model, "model")
            print(f"Tuned RF - n: {n}, depth: {depth}, acc: {acc:.4f}")
