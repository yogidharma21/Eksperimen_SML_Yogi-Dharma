import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# ============================================================
# automate_Yogi-Dharma.py
# Otomatisasi preprocessing dataset Credit Card Fraud
# Author: Yogi-Dharma
# ============================================================


def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Parameters
    ----------
    filepath : str
        Path ke file CSV raw dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame hasil loading.
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak diperlukan untuk pelatihan model.
    TransactionID dihapus karena hanya berfungsi sebagai identifier unik.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa kolom yang tidak perlu.
    """
    cols_to_drop = ["TransactionID"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"[INFO] Kolom dihapus: {cols_to_drop}")
    return df


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengekstrak fitur-fitur dari kolom TransactionDate:
    - transaction_hour   : jam transaksi (0-23)
    - transaction_day    : hari dalam bulan (1-31)
    - transaction_month  : bulan (1-12)
    - transaction_dayofweek : hari dalam minggu (0=Senin, 6=Minggu)

    Kolom TransactionDate asli kemudian dihapus.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur tanggal/waktu baru.
    """
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["transaction_hour"] = df["TransactionDate"].dt.hour
    df["transaction_day"] = df["TransactionDate"].dt.day
    df["transaction_month"] = df["TransactionDate"].dt.month
    df["transaction_dayofweek"] = df["TransactionDate"].dt.dayofweek
    df = df.drop(columns=["TransactionDate"])
    print("[INFO] Fitur datetime diekstrak: hour, day, month, dayofweek")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding pada kolom kategorikal:
    - TransactionType : Binary encoding (purchase=1, refund=0)
    - Location        : Label Encoding (10 kota unik)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom kategorikal sudah diencoding.
    """
    # Binary encoding untuk TransactionType
    if "TransactionType" in df.columns:
        df["TransactionType"] = df["TransactionType"].map({"purchase": 1, "refund": 0})
        print("[INFO] TransactionType diencoding: purchase=1, refund=0")

    # Label encoding untuk Location
    if "Location" in df.columns:
        le = LabelEncoder()
        df["Location"] = le.fit_transform(df["Location"])
        print(f"[INFO] Location diencoding dengan LabelEncoder: {list(le.classes_)}")

    return df


def scale_features(df: pd.DataFrame, target_col: str = "IsFraud") -> pd.DataFrame:
    """
    Melakukan StandardScaler pada fitur numerik (kecuali target).
    Kolom yang di-scale: Amount, MerchantID, transaction_hour,
    transaction_day, transaction_month, transaction_dayofweek.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.
    target_col : str
        Nama kolom target yang tidak akan di-scale.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur numerik sudah di-scale.
    """
    cols_to_scale = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"[INFO] StandardScaler diterapkan pada: {cols_to_scale}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values jika ada.
    - Numerik  : diisi dengan median kolom
    - Kategorikal : diisi dengan modus kolom

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa missing values.
    """
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
    """
    Pipeline utama preprocessing yang menggabungkan semua tahapan:
    1. Load data
    2. Drop kolom tidak perlu
    3. Handle missing values
    4. Ekstrak fitur datetime
    5. Encode kategorikal
    6. Scale fitur numerik

    Parameters
    ----------
    input_path : str
        Path ke file CSV raw dataset.
    output_path : str, optional
        Path untuk menyimpan hasil preprocessing. Jika None, tidak disimpan.

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah siap dilatih.
    """
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


# ============================================================
# Entry point
# ============================================================
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
