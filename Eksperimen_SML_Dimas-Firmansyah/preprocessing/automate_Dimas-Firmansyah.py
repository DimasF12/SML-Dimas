import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_heart_data(input_path, output_path):
    # 1. Load dataset
    df = pd.read_csv(input_path)

    # 2. Encode kolom kategorikal
    le = LabelEncoder()
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # 3. Scaling kolom numerik
    scaler = StandardScaler()
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 4. Simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing selesai! File disimpan di: {output_path}")

# Eksekusi saat file dijalankan langsung
if __name__ == '__main__':
    input_path = '../heart_raw/heart.csv'
    output_path = 'heart_preprocessing/heart_processed.csv'
    preprocess_heart_data(input_path, output_path)
