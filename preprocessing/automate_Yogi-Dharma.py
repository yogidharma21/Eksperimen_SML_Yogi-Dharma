import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["TransactionID"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"[INFO] Kolom dihapus: {cols_to_drop}")
    return df


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["transaction_hour"] = df["TransactionDate"].dt.hour
    df["transaction_day"] = df["TransactionDate"].dt.day
    df["transaction_month"] = df["TransactionDate"].dt.month
    df["transaction_dayofweek"] = df["TransactionDate"].dt.dayofweek
    df = df.drop(columns=["TransactionDate"])
    print("[INFO] Fitur datetime diekstrak: hour, day, month, dayofweek")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionType" in df.columns:
        df["TransactionType"] = df["TransactionType"].map({"purchase": 1, "refund": 0})
        print("[INFO] TransactionType diencoding: purchase=1, refund=0")

    if "Location" in df.columns:
        le = LabelEncoder()
        df["Location"] = le.fit_transform(df["Location"])
        print(f"[INFO] Location diencoding dengan LabelEncoder: {list(le.classes_)}")

    return df


def scale_features(df: pd.DataFrame, target_col: str = "IsFraud") -> pd.DataFrame:
    cols_to_scale = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"[INFO] StandardScaler diterapkan pada: {cols_to_scale}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_before = df.isnull().sum().sum()
    if missing_before == 0:
        print("[INFO] Tidak ada missing values ditemukan.")
        return df

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"[INFO] Missing values ditangani: {missing_before} nilai diisi.")
    return df


def preprocess(input_path: str, output_path: str = None) -> pd.DataFrame:
    print("=" * 55)
    print("  PIPELINE PREPROCESSING — Credit Card Fraud Dataset")
    print("  Author: Yogi-Dharma")
    print("=" * 55)

    df = load_data(input_path)
    df = drop_unnecessary_columns(df)
    df = handle_missing_values(df)
    df = extract_datetime_features(df)
    df = encode_categorical(df)
    df = scale_features(df, target_col="IsFraud")

    print("-" * 55)
    print(f"[DONE] Preprocessing selesai. Shape akhir: {df.shape}")
    print(f"[INFO] Kolom akhir: {df.columns.tolist()}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SAVE] Dataset tersimpan di: {output_path}")

    print("=" * 55)
    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(BASE_DIR, "credit_card_fraud_dataset_raw.csv")
    OUTPUT_PATH = os.path.join(
        BASE_DIR,
        "preprocessing",
        "credit_card_fraud_dataset_preprocessing",
        "credit_card_fraud_preprocessing.csv",
    )

    df_ready = preprocess(input_path=INPUT_PATH, output_path=OUTPUT_PATH)
    print(df_ready.head())
