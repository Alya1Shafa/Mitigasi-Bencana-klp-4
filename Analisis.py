import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB 
from sklearn.linear_model import LogisticRegression # Model Kedua
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE # Teknik Over-sampling

# ⚠️ GANTI DENGAN NAMA 3 FILE CSV ANDA! ⚠️
FILE_NAMES = [
    'banjir_data_realtime.csv',   
    'longsor_data_realtime.csv',  
    'gempa_data_realtime.csv'     
]
# =========================================================
# rendii 21.42
# --- DAFTAR STOPWORDS BAHASA INDONESIA ---
INDO_STOPWORDS = [
    'yang', 'untuk', 'pada', 'saya', 'dan', 'adalah', 'di', 'ke', 'dari', 
    'tidak', 'dengan', 'ini', 'itu', 'kita', 'mereka', 'semua', 'ada', 
    'akan', 'bisa', 'sudah', 'telah', 'saat', 'hanya', 'iya', 'ya', 'gak', 
    'nya', 'sih', 'tetap', 'tapi', 'kalau', 'udah', 'lagi', 'pun', 'agar'
]

# --- 1. Fungsi Pelabelan Sentimen (Simulasi Bahasa Indonesia) ---
def simple_sentiment_labeling(text):
    text = str(text).lower()
    if 'rusak' in text or 'korban' in text or 'parah' in text or 'terjebak' in text or 'butuh' in text or 'bahaya' in text or 'bantu' in text or 'takut' in text:
        return 'Negative'
    elif 'selamat' in text or 'syukur' in text or 'aman' in text or 'pulih' in text or 'doa' in text or 'terima kasih' in text:
        return 'Positive'
    else:
        return 'Neutral'

# --- 2. Menggabungkan Data ---
all_data = []
for file in FILE_NAMES:
    try:
        df_temp = pd.read_csv(file)
        df_temp['Disaster_Type'] = file.split('_data')[0]
        all_data.append(df_temp)
    except FileNotFoundError:
        print(f"Peringatan: File {file} tidak ditemukan. Melewati file ini.")
        
if not all_data:
    print("ERROR: Tidak ada file data yang berhasil dimuat. Program berhenti.")
    sys.exit(1)

df = pd.concat(all_data, ignore_index=True)
df_filtered = df.copy() 

# Pembersihan Data
df_filtered['Text'] = df_filtered['Text'].astype(str).replace(r'^\s*$', np.nan, regex=True)
df_filtered.dropna(subset=['Text'], inplace=True)

# Pelabelan Sentimen
df_filtered['Sentiment'] = df_filtered['Text'].apply(simple_sentiment_labeling)

# --- 3. Persiapan Data dan Vectorization ---
X = df_filtered['Text'] 
y = df_filtered['Sentiment'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000, stop_words=INDO_STOPWORDS, ngram_range=(1, 2)) 

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 4. Mengatasi Ketidakseimbangan dengan SMOTE ---
print("\n[SMOTE] Melakukan Over-sampling pada data pelatihan...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)

print(f"Distribusi data pelatihan Awal: {y_train.value_counts().to_markdown()}")
print(f"Distribusi data pelatihan Setelah SMOTE: {y_train_smote.value_counts().to_markdown()}")


# --- 5. Pelatihan dan Evaluasi Model 1: Complement Naive Bayes (CNB) ---
model_cnb = ComplementNB()
model_cnb.fit(X_train_smote, y_train_smote) # Gunakan data yang sudah diseimbangkan
y_pred_cnb = model_cnb.predict(X_test_vec)
accuracy_cnb = accuracy_score(y_test, y_pred_cnb)

# --- 6. Pelatihan dan Evaluasi Model 2: Logistic Regression (LR) ---
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train_smote, y_train_smote) # Gunakan data yang sudah diseimbangkan
y_pred_lr = model_lr.predict(X_test_vec)
accuracy_lr = accuracy_score(y_test, y_pred_lr)


# --- 7. Tampilkan Hasil Perbandingan ---
print("\n" + "="*80)
print("             LAPORAN ANALISIS SENTIMEN TINGKAT LANJUT (SMOTE & PERBANDINGAN MODEL)")
print("="*80)

# Laporan CNB
print("\n--- Model 1: Complement Naive Bayes (Setelah SMOTE) ---")
print(f"Akurasi: {accuracy_cnb:.4f}")
print(classification_report(y_test, y_pred_cnb, zero_division=0))

# Laporan LR
print("\n--- Model 2: Logistic Regression (Setelah SMOTE) ---")
print(f"Akurasi: {accuracy_lr:.4f}")
print(classification_report(y_test, y_pred_lr, zero_division=0))

print("\nKESIMPULAN:")
if accuracy_lr > accuracy_cnb:
    print(f"Model Logistic Regression (Akurasi: {accuracy_lr:.4f}) lebih unggul dalam memprediksi data uji.")
else:
    print(f"Model Complement Naive Bayes (Akurasi: {accuracy_cnb:.4f}) lebih unggul dalam memprediksi data uji.")


print("==================================================")







