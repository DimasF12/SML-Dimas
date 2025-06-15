import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 Akurasi: {acc:.4f}")
    return acc

def main(data_path):
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(data_path)

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        acc = evaluate_model(model, X_test, y_test)

        # Simpan model secara eksplisit (opsional karena autolog sudah menyimpan juga)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="heart_processed.csv")
    args = parser.parse_args()

    main(args.data_path)
