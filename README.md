# Eksperimen_SML_Yogi-Dharma

> Submission Machine Learning — Eksperimen Preprocessing Dataset  
> **Nama:** Yogi-Dharma  
> **GitHub:** [yogidharma21](https://github.com/yogidharma21)  
> **Repository:** [Eksperimen_SML_Yogi-Dharma](https://github.com/yogidharma21/Eksperimen_SML_Yogi-Dharma)  
> **Dataset:** Credit Card Fraud Detection  

---

## 📁 Struktur Repository

```
Eksperimen_SML_Yogi-Dharma/
├── .github/
│   └── workflows/
│       └── preprocessing.yml               ← GitHub Actions (Advance)
├── credit_card_fraud_dataset_raw.csv       ← Dataset mentah
└── preprocessing/
    ├── Eksperimen_Yogi-Dharma.ipynb        ← Notebook eksperimen (EDA + preprocessing)
    ├── automate_Yogi-Dharma.py             ← Script otomatisasi preprocessing (Skilled)
    └── credit_card_fraud_dataset_preprocessing/
        └── credit_card_fraud_preprocessing.csv  ← Hasil preprocessing
```

---

## 📊 Tentang Dataset

| Info | Detail |
|---|---|
| **Nama** | Credit Card Fraud Detection |
| **Jumlah Baris** | 100.000 transaksi |
| **Jumlah Kolom** | 7 (raw) → 9 (setelah preprocessing) |
| **Target** | `IsFraud` (0 = tidak fraud, 1 = fraud) |
| **Imbalance** | 99% tidak fraud, 1% fraud |

### Kolom Dataset Raw
| Kolom | Tipe | Keterangan |
|---|---|---|
| `TransactionID` | int | ID unik transaksi |
| `TransactionDate` | datetime | Waktu transaksi |
| `Amount` | float | Nominal transaksi |
| `MerchantID` | int | ID merchant |
| `TransactionType` | string | `purchase` / `refund` |
| `Location` | string | Kota transaksi |
| `IsFraud` | int | Label (0/1) |

---

## ⚙️ Tahapan Preprocessing

| # | Tahap | Tindakan |
|---|---|---|
| 1 | **Drop Kolom** | Hapus `TransactionID` — hanya identifier, tidak prediktif |
| 2 | **Missing Values** | Deteksi otomatis; numerik → median, kategorikal → modus |
| 3 | **Datetime Extraction** | Ekstrak `transaction_hour`, `transaction_day`, `transaction_month`, `transaction_dayofweek` dari `TransactionDate` |
| 4 | **Encoding** | `TransactionType` → binary (purchase=1, refund=0); `Location` → Label Encoding |
| 5 | **Feature Scaling** | StandardScaler pada semua fitur kecuali `IsFraud` |

### Kolom Dataset Setelah Preprocessing
`Amount`, `MerchantID`, `TransactionType`, `Location`, `IsFraud`, `transaction_hour`, `transaction_day`, `transaction_month`, `transaction_dayofweek`

---

## 🤖 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/yogidharma21/Eksperimen_SML_Yogi-Dharma.git
cd Eksperimen_SML_Yogi-Dharma
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 3. Jalankan Script Otomatis
```bash
python preprocessing/automate_Yogi-Dharma.py
```
Output akan tersimpan di `preprocessing/credit_card_fraud_dataset_preprocessing/credit_card_fraud_preprocessing.csv`

### 4. Jalankan Notebook
```bash
pip install jupyter
jupyter notebook preprocessing/Eksperimen_Yogi-Dharma.ipynb
```

---

## 🔄 GitHub Actions

Workflow otomatis akan berjalan ketika:
- Ada **push** pada file `credit_card_fraud_dataset_raw.csv` atau `automate_Yogi-Dharma.py`
- **Manual trigger** via Actions tab di GitHub

Hasil preprocessing akan di-commit otomatis ke repository.

---

## 🛠️ Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```
