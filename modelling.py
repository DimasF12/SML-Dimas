import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# Autolog
mlflow.sklearn.autolog()

# Set experiment name
mlflow.set_experiment("HeartDisease_Classifier")

# Load dataset
df = pd.read_csv('heart_processed.csv')

# Split data
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run MLflow experiment
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 Akurasi Model: {acc:.4f}")

    # Tambah signature dan contoh input
    signature = infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_test.iloc[:2],
        signature=signature
    )
